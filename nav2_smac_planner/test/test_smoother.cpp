// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <utility>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/smoother.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace nav2_smac_planner;  // NOLINT

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

class SmootherWrapper : public nav2_smac_planner::Smoother
{
public:
  explicit SmootherWrapper(const SmootherParams & params)
  : nav2_smac_planner::Smoother(params)
  {}

  std::vector<PathSegment> findDirectionalPathSegmentsWrapper(nav_msgs::msg::Path path)
  {
    return findDirectionalPathSegments(path);
  }
};

TEST(SmootherTest, rejects_non_circular_footprint_collision)
{
  SmootherParams params;
  params.tolerance_ = 2.0;
  params.max_its_ = 1000;
  params.w_data_ = 0.2;
  params.w_smooth_ = 0.3;
  params.do_refinement_ = false;
  params.holonomic_ = false;
  SmootherWrapper smoother(params);
  smoother.initialize(0.4);

  nav2_costmap_2d::Costmap2D costmap(
    200, 200, 0.1, 0.0, 0.0, nav2_costmap_2d::FREE_SPACE);
  nav2_costmap_2d::Footprint footprint;
  for (const auto & coordinates : std::vector<std::pair<double, double>>{
    {-0.6, -0.2}, {0.6, -0.2}, {0.6, 0.2}, {-0.6, 0.2}})
  {
    geometry_msgs::msg::Point point;
    point.x = coordinates.first;
    point.y = coordinates.second;
    footprint.push_back(point);
  }
  // The original path clears this obstacle; pulling the corner inward rotates
  // the long rectangular footprint into it while the center cell remains free.
  costmap.setCost(31, 117, nav2_costmap_2d::LETHAL_OBSTACLE);
  nav_msgs::msg::Path original;
  geometry_msgs::msg::PoseStamped pose;
  for (unsigned int i = 0; i <= 5; ++i) {
    pose.pose.position.x = 2.0;
    pose.pose.position.y = 2.0 + 2.0 * i;
    pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(
      i == 5 ? 0.0 : M_PI_2);
    original.poses.push_back(pose);
  }
  for (unsigned int i = 1; i <= 6; ++i) {
    pose.pose.position.x = 2.0 + 2.0 * i;
    pose.pose.position.y = 12.0;
    pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(0.0);
    original.poses.push_back(pose);
  }
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(&costmap);
  const auto collides = [&](const nav_msgs::msg::Path & path) {
      for (const auto & stamped : path.poses) {
        const auto & p = stamped.pose;
        if (checker.footprintCostAtPose(p.position.x, p.position.y,
            tf2::getYaw(p.orientation), footprint) >= nav2_costmap_2d::LETHAL_OBSTACLE)
        {
          return true;
        }
      }
      return false;
    };
  ASSERT_FALSE(collides(original));

  auto center_only = original;
  ASSERT_TRUE(smoother.smooth(center_only, &costmap, 1.0));
  EXPECT_TRUE(collides(center_only));

  auto footprint_checked = original;
  EXPECT_FALSE(smoother.smooth(footprint_checked, &costmap, 1.0, footprint));
  EXPECT_FALSE(collides(footprint_checked));
}

TEST(SmootherTest, test_full_smoother)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr node =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacSmootherTest");
  nav2_smac_planner::SmootherParams params;
  params.get(node, "test");
  double maxtime = 1.0;

  // Make smoother and costmap to smooth in
  auto smoother = std::make_unique<SmootherWrapper>(params);
  smoother->initialize(0.4 /*turning radius*/);

  nav2_costmap_2d::Costmap2D * costmap =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.05, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 20; i <= 30; ++i) {
    for (unsigned int j = 20; j <= 30; ++j) {
      costmap->setCost(i, j, 254);
    }
  }

  // Setup A* search to get path to smooth
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.05;
  info.non_straight_penalty = 1.05;
  info.reverse_penalty = 2.0;
  info.cost_penalty = 2.0;
  info.retrospective_penalty = 0.0;
  info.analytic_expansion_ratio = 3.5;
  info.minimum_turning_radius = 8;  // in grid coordinates 0.4/0.05
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  unsigned int size_theta = 72;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, std::numeric_limits<int>::max(), max_planning_time, 401, size_theta);
  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap, size_theta, node);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // Create A* search to smooth
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(5u, 5u, 0u);
  a_star.setGoal(45u, 45u, 36u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance));

  // Convert to world coordinates and get length to compare to smoothed length
  nav_msgs::msg::Path plan;
  plan.header.stamp = node->now();
  plan.header.frame_id = "map";
  geometry_msgs::msg::PoseStamped pose;
  pose.header = plan.header;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = 0.0;
  pose.pose.orientation.w = 1.0;
  double initial_length = 0.0;
  double x_m = path[path.size() - 1].x, y_m = path[path.size() - 1].y;
  plan.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.pose = nav2_smac_planner::getWorldCoords(path[i].x, path[i].y, costmap);
    pose.pose.orientation = nav2_smac_planner::getWorldOrientation(path[i].theta);
    plan.poses.push_back(pose);
    initial_length += hypot(path[i].x - x_m, path[i].y - y_m);
    x_m = path[i].x;
    y_m = path[i].y;
  }

  // Check that we accurately detect that this path has a reversing segment
  EXPECT_EQ(smoother->findDirectionalPathSegmentsWrapper(plan).size(), 2u);

  // Test smoother, should succeed with same number of points
  // and shorter overall length, while still being collision free.
  auto path_size_in = plan.poses.size();
  EXPECT_TRUE(smoother->smooth(plan, costmap, maxtime));
  EXPECT_EQ(plan.poses.size(), path_size_in);  // Should have same number of poses
  double length = 0.0;
  x_m = plan.poses[0].pose.position.x;
  y_m = plan.poses[0].pose.position.y;
  for (unsigned int i = 0; i != plan.poses.size(); i++) {
    // Should be collision free
    EXPECT_EQ(costmap->getCost(plan.poses[i].pose.position.x, plan.poses[i].pose.position.y), 0);
    length += hypot(plan.poses[i].pose.position.x - x_m, plan.poses[i].pose.position.y - y_m);
    x_m = plan.poses[i].pose.position.x;
    y_m = plan.poses[i].pose.position.y;
  }
  EXPECT_LT(length, initial_length);  // Should be shorter

  // Try again but with failure modes

  // Failure mode: not enough iterations to complete
  params.max_its_ = 0;
  auto smoother_bypass = std::make_unique<SmootherWrapper>(params);
  EXPECT_FALSE(smoother_bypass->smooth(plan, costmap, maxtime));
  params.max_its_ = 1;
  auto smoother_failure = std::make_unique<SmootherWrapper>(params);
  EXPECT_FALSE(smoother_failure->smooth(plan, costmap, maxtime));

  // Failure mode: Not enough time
  double max_no_time = 0.0;
  EXPECT_FALSE(smoother->smooth(plan, costmap, max_no_time));

  // Failure mode: Path is in collision, do 2x to exercise overlapping point
  // attempts to update orientation should also fail
  pose.pose.position.x = 1.25;
  pose.pose.position.y = 1.25;
  plan.poses.push_back(pose);
  plan.poses.push_back(pose);
  EXPECT_FALSE(smoother->smooth(plan, costmap, maxtime));
  EXPECT_NEAR(plan.poses.end()[-2].pose.orientation.z, 1.0, 1e-3);
  EXPECT_NEAR(plan.poses.end()[-2].pose.orientation.x, 0.0, 1e-3);
  EXPECT_NEAR(plan.poses.end()[-2].pose.orientation.y, 0.0, 1e-3);
  EXPECT_NEAR(plan.poses.end()[-2].pose.orientation.w, 0.0, 1e-3);

  delete costmap;
}
