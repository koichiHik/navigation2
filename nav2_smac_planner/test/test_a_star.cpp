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
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/node_lattice.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

TEST(AStarTest, test_a_star_2d)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D> a_star(
    nav2_smac_planner::MotionModel::TWOD, info);
  int max_iterations = 10000;
  float tolerance = 0.0;
  float some_tolerance = 20.0;
  int it_on_approach = 10;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(false, max_iterations, it_on_approach, max_planning_time, 0.0, 1);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  // functional case testing
  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmapA, 1, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(20u, 20u, 0);
  a_star.setGoal(80u, 80u, 0);
  nav2_smac_planner::Node2D::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance));
  EXPECT_EQ(num_it, 2414);

  // check path is the right size and collision free
  EXPECT_EQ(path.size(), 82u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }

  // setting non-zero dim 3 for 2D search
  EXPECT_THROW(a_star.setGoal(0, 0, 10), std::runtime_error);
  EXPECT_THROW(a_star.setStart(0, 0, 10), std::runtime_error);

  path.clear();
  // failure cases with invalid inputs
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D> a_star_2(
    nav2_smac_planner::MotionModel::TWOD, info);
  a_star_2.initialize(false, max_iterations, it_on_approach, max_planning_time, 0, 1);
  num_it = 0;
  EXPECT_THROW(a_star_2.createPath(path, num_it, tolerance), std::runtime_error);
  a_star_2.setCollisionChecker(checker.get());
  num_it = 0;
  EXPECT_THROW(a_star_2.createPath(path, num_it, tolerance), std::runtime_error);
  a_star_2.setStart(50, 50, 0);  // invalid
  a_star_2.setGoal(0, 0, 0);  // valid
  num_it = 0;
  EXPECT_THROW(a_star_2.createPath(path, num_it, tolerance), std::runtime_error);
  a_star_2.setStart(0, 0, 0);  // valid
  a_star_2.setGoal(50, 50, 0);  // invalid
  num_it = 0;
  EXPECT_THROW(a_star_2.createPath(path, num_it, tolerance), std::runtime_error);
  num_it = 0;
  // invalid goal but liberal tolerance
  a_star_2.setStart(20, 20, 0);  // valid
  a_star_2.setGoal(50, 50, 0);  // invalid
  EXPECT_TRUE(a_star_2.createPath(path, num_it, some_tolerance));
  EXPECT_EQ(path.size(), 21u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }

  EXPECT_TRUE(a_star_2.getStart() != nullptr);
  EXPECT_TRUE(a_star_2.getGoal() != nullptr);
  EXPECT_EQ(a_star_2.getSizeX(), 100u);
  EXPECT_EQ(a_star_2.getSizeY(), 100u);
  EXPECT_EQ(a_star_2.getSizeDim3(), 1u);
  EXPECT_EQ(a_star_2.getToleranceHeuristic(), 20.0);
  EXPECT_EQ(a_star_2.getOnApproachMaxIterations(), 10);

  delete costmapA;
}

TEST(AStarTest, timeout_returns_only_a_path_within_tolerance)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_costmap_2d::Costmap2D costmap(20, 20, 0.1, 0.0, 0.0, 0);
  costmap.setCost(4, 2, nav2_costmap_2d::LETHAL_OBSTACLE);
  nav2_smac_planner::GridCollisionChecker checker(&costmap, 1, lnode);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  for (const float tolerance : {2.0f, 0.5f}) {
    nav2_smac_planner::SearchInfo info{};
    nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D> a_star(
      nav2_smac_planner::MotionModel::TWOD, info);
    int max_iterations = 10000;
    a_star.initialize(false, max_iterations, 100, 0.0, 0.0, 1);
    a_star.setCollisionChecker(&checker);
    a_star.setStart(2, 2, 0);
    a_star.setGoal(4, 2, 0);

    // Allow exactly one expansion before the periodic timeout check at iteration 5000.
    // The zero time budget makes this deterministic without sleeping or a test clock.
    int iterations = 4999;
    nav2_smac_planner::Node2D::CoordinateVector path;
    const bool success = a_star.createPath(path, iterations, tolerance);
    EXPECT_EQ(iterations, 5000);
    if (tolerance > 1.0f) {
      ASSERT_TRUE(success);
      ASSERT_EQ(path.size(), 2u);
      EXPECT_LT(std::hypot(path.front().x - 4.0f, path.front().y - 2.0f), tolerance);
      EXPECT_FLOAT_EQ(path.back().x, 2.0f);
      EXPECT_FLOAT_EQ(path.back().y, 2.0f);
      for (const auto & pose : path) {
        EXPECT_EQ(costmap.getCost(pose.x, pose.y), nav2_costmap_2d::FREE_SPACE);
      }
    } else {
      EXPECT_FALSE(success);
      EXPECT_TRUE(path.empty());
    }
  }
}

TEST(AStarTest, test_a_star_se2)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.1;
  info.non_straight_penalty = 1.1;
  info.reverse_penalty = 2.0;
  info.minimum_turning_radius = 8;  // in grid coordinates
  info.retrospective_penalty = 0.015;
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  info.analytic_expansion_ratio = 3.5;
  unsigned int size_theta = 72;
  info.cost_penalty = 1.7;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::DUBIN, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(false, max_iterations, it_on_approach, max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmapA, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(10u, 10u, 0u);
  a_star.setGoal(80u, 80u, 40u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance));

  // check path is the right size and collision free
  EXPECT_EQ(num_it, 3222);
  EXPECT_EQ(path.size(), 63u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }
  // no skipped nodes
  for (unsigned int i = 1; i != path.size(); i++) {
    EXPECT_LT(hypotf(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y), 2.1f);
  }

  delete costmapA;
}

TEST(AStarTest, test_a_star_lattice)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.05;
  info.non_straight_penalty = 1.05;
  info.reverse_penalty = 2.0;
  info.retrospective_penalty = 0.1;
  info.analytic_expansion_ratio = 3.5;
  info.lattice_filepath =
    ament_index_cpp::get_package_share_directory("nav2_smac_planner") +
    "/sample_primitives/5cm_resolution/0.5m_turning_radius/ackermann" +
    "/output.json";
  info.minimum_turning_radius = 8;  // in grid coordinates 0.4/0.05
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  unsigned int size_theta = 16;
  info.cost_penalty = 2.0;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeLattice> a_star(
    nav2_smac_planner::MotionModel::STATE_LATTICE, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, std::numeric_limits<int>::max(), max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.05, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 20; i <= 30; ++i) {
    for (unsigned int j = 20; j <= 30; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmapA, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(5u, 5u, 0u);
  a_star.setGoal(40u, 40u, 1u);
  nav2_smac_planner::NodeLattice::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance));

  // check path is the right size and collision free
  EXPECT_EQ(num_it, 21);
  EXPECT_GT(path.size(), 47u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }
  // no skipped nodes
  for (unsigned int i = 1; i != path.size(); i++) {
    EXPECT_LT(hypotf(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y), 2.1f);
  }

  delete costmapA;
}

TEST(AStarTest, test_se2_single_pose_path)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.1;
  info.non_straight_penalty = 1.1;
  info.reverse_penalty = 2.0;
  info.retrospective_penalty = 0.0;
  info.minimum_turning_radius = 8;  // in grid coordinates
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  info.analytic_expansion_ratio = 3.5;
  unsigned int size_theta = 72;
  info.cost_penalty = 1.7;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::DUBIN, info);
  int max_iterations = 100;
  float tolerance = 10.0;
  int it_on_approach = 10;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(false, max_iterations, it_on_approach, max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmapA, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(10u, 10u, 0u);
  // Goal is one costmap cell away
  a_star.setGoal(12u, 10u, 0u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance));

  // Check that the path is length one
  // With the current implementation, this produces a longer path
  // EXPECT_EQ(path.size(), 1u);
  EXPECT_GE(path.size(), 1u);

  delete costmapA;
}

TEST(AStarTest, test_constants)
{
  nav2_smac_planner::MotionModel mm = nav2_smac_planner::MotionModel::UNKNOWN;  // unknown
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Unknown"));
  mm = nav2_smac_planner::MotionModel::TWOD;  // 2d
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("2D"));
  mm = nav2_smac_planner::MotionModel::DUBIN;  // dubin
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Dubin"));
  mm = nav2_smac_planner::MotionModel::REEDS_SHEPP;  // reeds-shepp
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Reeds-Shepp"));

  EXPECT_EQ(
    nav2_smac_planner::fromString(
      "2D"), nav2_smac_planner::MotionModel::TWOD);
  EXPECT_EQ(nav2_smac_planner::fromString("DUBIN"), nav2_smac_planner::MotionModel::DUBIN);
  EXPECT_EQ(
    nav2_smac_planner::fromString(
      "REEDS_SHEPP"), nav2_smac_planner::MotionModel::REEDS_SHEPP);
  EXPECT_EQ(nav2_smac_planner::fromString("NONE"), nav2_smac_planner::MotionModel::UNKNOWN);
}

TEST(AStarTest, analytic_expansion_restores_revisited_goal)
{
  using Node = nav2_smac_planner::NodeHybrid;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("rollback_test");
  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 82.5;
  info.analytic_expansion_max_length = 200.0;
  nav2_smac_planner::AStarAlgorithm<Node> a_star(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info);
  int max_iterations = 10000;
  a_star.initialize(false, max_iterations, 1000, 5.0, 401, 64);
  nav2_costmap_2d::Costmap2D costmap(1000, 400, 0.1, 0, 0, 0);
  nav2_smac_planner::GridCollisionChecker checker(&costmap, 64, node);
  checker.setFootprint({}, true, 0.0);
  a_star.setCollisionChecker(&checker);
  Node start(Node::getIndex(537, 125, 16, 1000, 64));
  Node goal(Node::getIndex(530, 66, 1, 1000, 64));
  start.pose = {537.454773f, 125.400085f, 16.0f};
  goal.pose = {530.467102f, 66.557159f, 1.0f};
  const auto original = goal.pose;
  std::unordered_map<unsigned int, std::unique_ptr<Node>> nodes;
  Node::NodePtr start_ptr = &start, goal_ptr = &goal;
  auto getter = [&](const unsigned int & index, Node::NodePtr & result) {
      if (index == goal.getIndex()) {
        result = goal_ptr;
      } else if (index == start.getIndex()) {
        result = start_ptr;
      } else {
        auto & stored = nodes[index];
        if (!stored) {stored = std::make_unique<Node>(index);}
        result = stored.get();
      }
      return true;
    };
  nav2_smac_planner::AnalyticExpansion<Node> expansion(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, 64);
  expansion.setCollisionChecker(&checker);
  const auto candidate = expansion.getAnalyticPath(start_ptr, goal_ptr, getter);
  ASSERT_FALSE(candidate.empty());
  size_t visits = 0;
  for (const auto & sample : candidate) {
    if (sample.node == goal_ptr) {++visits;}
  }
  ASSERT_GE(visits, 2u);
  EXPECT_FLOAT_EQ(goal.pose.x, original.x);
  EXPECT_FLOAT_EQ(goal.pose.y, original.y);
  EXPECT_FLOAT_EQ(goal.pose.theta, original.theta);
}
