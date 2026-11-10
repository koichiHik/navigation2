// Copyright (c) 2021 Samsung Research America
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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/smac_planner_lattice.hpp"
#include "nav2_core/exceptions.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

// Simple wrapper to be able to call a private member
class LatticeWrap : public nav2_smac_planner::SmacPlannerLattice
{
public:
  void callDynamicParams(std::vector<rclcpp::Parameter> parameters)
  {
    dynamicParametersCallback(parameters);
  }
};

class LatticeBoundsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("LatticeBoundsTest");
    costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>("bounds_costmap");
    const auto results = costmap_->set_parameters({
      rclcpp::Parameter("plugins", std::vector<std::string>{}),
      rclcpp::Parameter("resolution", 0.05), rclcpp::Parameter("width", 8),
      rclcpp::Parameter("height", 8), rclcpp::Parameter("origin_x", -4.0),
      rclcpp::Parameter("origin_y", -4.0)});
    for (const auto & result : results) {
      ASSERT_TRUE(result.successful) << result.reason;
    }
    const auto configured = costmap_->on_configure(rclcpp_lifecycle::State());
    costmap_configured_ = configured == nav2_util::CallbackReturn::SUCCESS;
    ASSERT_TRUE(costmap_configured_);
    fillFree();
    node_->declare_parameter("lattice.smooth_path", false);
    node_->declare_parameter("lattice.allow_unknown", false);
    node_->declare_parameter("lattice.tolerance", 0.05);
    node_->declare_parameter("lattice.lookup_table_size", 2.0);
    node_->declare_parameter("lattice.max_planning_time", 3.0);
    planner_ = std::make_unique<nav2_smac_planner::SmacPlannerLattice>();
    planner_->configure(node_, "lattice", costmap_->getTfBuffer(), costmap_);
  }

  void TearDown() override
  {
    if (planner_) {
      planner_->cleanup();
      planner_.reset();
    }
    if (costmap_configured_) {
      costmap_->on_cleanup(rclcpp_lifecycle::State());
    }
  }

  static geometry_msgs::msg::PoseStamped pose(double x, double y)
  {
    geometry_msgs::msg::PoseStamped result;
    result.header.frame_id = "map";
    result.pose.position.x = x;
    result.pose.position.y = y;
    result.pose.orientation.w = 1.0;
    return result;
  }

  void fillFree()
  {
    auto * map = costmap_->getCostmap();
    std::fill(map->getCharMap(),
      map->getCharMap() + map->getSizeInCellsX() * map->getSizeInCellsY(),
      nav2_costmap_2d::FREE_SPACE);
  }

  void expectValidPlan(const geometry_msgs::msg::PoseStamped & goal)
  {
    const auto path = planner_->createPlan(pose(0.025, 0.025), goal);
    ASSERT_GT(path.poses.size(), 1u);
    EXPECT_EQ(path.header.frame_id, "map");
    for (const auto & stamped : path.poses) {
      EXPECT_TRUE(std::isfinite(stamped.pose.position.x));
      EXPECT_TRUE(std::isfinite(stamped.pose.position.y));
    }
    EXPECT_NEAR(path.poses.back().pose.position.x, goal.pose.position.x, 0.1);
    EXPECT_NEAR(path.poses.back().pose.position.y, goal.pose.position.y, 0.1);
  }

  // All four edges, including the exact upper boundaries excluded by worldToMap().
  static std::vector<geometry_msgs::msg::PoseStamped> outsidePoses()
  {
    return {pose(-4.001, 0.025), pose(4.0, 0.025), pose(4.001, 0.025),
      pose(0.025, -4.001), pose(0.025, 4.0), pose(0.025, 4.001)};
  }

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  std::unique_ptr<nav2_smac_planner::SmacPlannerLattice> planner_;
  bool costmap_configured_{false};
};

TEST_F(LatticeBoundsTest, rejects_start_outside_each_map_edge)
{
  for (const auto & start : outsidePoses()) {
    SCOPED_TRACE(::testing::Message() << start.pose.position.x << "," << start.pose.position.y);
    EXPECT_THROW(planner_->createPlan(start, pose(1.025, 0.025)), nav2_core::PlannerException);
  }
}

TEST_F(LatticeBoundsTest, rejects_goal_outside_each_map_edge)
{
  for (const auto & goal : outsidePoses()) {
    SCOPED_TRACE(::testing::Message() << goal.pose.position.x << "," << goal.pose.position.y);
    EXPECT_THROW(planner_->createPlan(pose(0.025, 0.025), goal), nav2_core::PlannerException);
  }
}

TEST_F(LatticeBoundsTest, rolling_origin_rejects_old_goal_and_preserves_valid_plans)
{
  const auto old_goal = pose(1.025, 0.025);
  expectValidPlan(old_goal);
  costmap_->getCostmap()->updateOrigin(-7.0, -4.0);
  fillFree();
  EXPECT_THROW(planner_->createPlan(pose(0.025, 0.025), old_goal), nav2_core::PlannerException);
  expectValidPlan(pose(0.525, 0.025));
}

// SMAC smoke tests for plugin-level issues rather than algorithms
// (covered by more extensively testing in other files)
// System tests in nav2_system_tests will actually plan with this work

TEST(SmacTest, test_smac_lattice)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeLattice =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacLatticeTest");

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  geometry_msgs::msg::PoseStamped start, goal;
  start.pose.position.x = 0.0;
  start.pose.position.y = 0.0;
  start.pose.orientation.w = 1.0;
  goal.pose.position.x = 1.0;
  goal.pose.position.y = 1.0;
  goal.pose.orientation.w = 1.0;
  auto planner = std::make_unique<nav2_smac_planner::SmacPlannerLattice>();
  try {
    // Expect to throw due to invalid prims file in param
    planner->configure(nodeLattice, "test", nullptr, costmap_ros);
  } catch (...) {
  }
  planner->activate();

  try {
    planner->createPlan(start, goal);
  } catch (...) {
  }

  planner->deactivate();
  planner->cleanup();

  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
  costmap_ros.reset();
  nodeLattice.reset();
}

TEST(SmacTest, test_smac_lattice_reconfigure)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeLattice =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacLatticeTest");

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  auto planner = std::make_unique<LatticeWrap>();
  try {
    // Expect to throw due to invalid prims file in param
    planner->configure(nodeLattice, "test", nullptr, costmap_ros);
  } catch (...) {
  }
  planner->activate();

  auto rec_param = std::make_shared<rclcpp::AsyncParametersClient>(
    nodeLattice->get_node_base_interface(), nodeLattice->get_node_topics_interface(),
    nodeLattice->get_node_graph_interface(),
    nodeLattice->get_node_services_interface());

  auto results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.allow_unknown", false),
      rclcpp::Parameter("test.max_iterations", -1),
      rclcpp::Parameter("test.cache_obstacle_heuristic", true),
      rclcpp::Parameter("test.reverse_penalty", 5.0),
      rclcpp::Parameter("test.change_penalty", 1.0),
      rclcpp::Parameter("test.non_straight_penalty", 2.0),
      rclcpp::Parameter("test.cost_penalty", 2.0),
      rclcpp::Parameter("test.retrospective_penalty", 0.2),
      rclcpp::Parameter("test.analytic_expansion_ratio", 4.0),
      rclcpp::Parameter("test.max_planning_time", 10.0),
      rclcpp::Parameter("test.lookup_table_size", 30.0),
      rclcpp::Parameter("test.smooth_path", false),
      rclcpp::Parameter("test.analytic_expansion_max_length", 42.0),
      rclcpp::Parameter("test.tolerance", 42.0),
      rclcpp::Parameter("test.rotation_penalty", 42.0),
      rclcpp::Parameter("test.max_on_approach_iterations", 42),
      rclcpp::Parameter("test.allow_reverse_expansion", true)});

  try {
    // All of these params will re-init A* which will involve loading the control set file
    // which will cause an exception because the file does not exist. This will cause an
    // expected failure preventing parameter updates from being successfully processed
    rclcpp::spin_until_future_complete(
      nodeLattice->get_node_base_interface(),
      results);
  } catch (...) {
  }

  // So instead, lets call manually on a change
  std::vector<rclcpp::Parameter> parameters;
  parameters.push_back(rclcpp::Parameter("test.lattice_filepath", std::string("HI")));
  EXPECT_THROW(planner->callDynamicParams(parameters), std::runtime_error);
}
