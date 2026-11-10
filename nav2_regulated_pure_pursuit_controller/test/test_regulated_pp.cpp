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
// limitations under the License.

#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <thread>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "path_utils/path_utils.hpp"
#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_core/goal_checker.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

class BasicAPIRPP : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  BasicAPIRPP()
  : nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController() {}

  nav_msgs::msg::Path getPlan() {return global_plan_;}

  double getSpeed() {return desired_linear_vel_;}

  double getGoalTolerance() {return goal_dist_tol_;}

  rcl_interfaces::msg::SetParametersResult updateParameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    return dynamicParametersCallback(parameters);
  }

  std::unique_ptr<geometry_msgs::msg::PointStamped> createCarrotMsgWrapper(
    const geometry_msgs::msg::PoseStamped & carrot_pose)
  {
    return createCarrotMsg(carrot_pose);
  }

  void setVelocityScaledLookAhead() {use_velocity_scaled_lookahead_dist_ = true;}
  void setCostRegulationScaling() {use_cost_regulated_linear_velocity_scaling_ = true;}
  void resetVelocityRegulationScaling() {use_regulated_linear_velocity_scaling_ = false;}

  double getLookAheadDistanceWrapper(const geometry_msgs::msg::Twist & twist)
  {
    return getLookAheadDistance(twist);
  }

  static geometry_msgs::msg::Point circleSegmentIntersectionWrapper(
    const geometry_msgs::msg::Point & p1,
    const geometry_msgs::msg::Point & p2,
    double r)
  {
    return circleSegmentIntersection(p1, p2, r);
  }

  geometry_msgs::msg::PoseStamped getLookAheadPointWrapper(
    const double & dist, const nav_msgs::msg::Path & path)
  {
    return getLookAheadPoint(dist, path);
  }

  bool shouldRotateToPathWrapper(
    const geometry_msgs::msg::PoseStamped & carrot_pose, double & angle_to_path)
  {
    return shouldRotateToPath(carrot_pose, angle_to_path);
  }

  bool shouldRotateToGoalHeadingWrapper(const geometry_msgs::msg::PoseStamped & carrot_pose)
  {
    return shouldRotateToGoalHeading(carrot_pose);
  }

  void rotateToHeadingWrapper(
    double & linear_vel, double & angular_vel,
    const double & angle_to_path, const geometry_msgs::msg::Twist & curr_speed)
  {
    return rotateToHeading(linear_vel, angular_vel, angle_to_path, curr_speed);
  }

  void applyConstraintsWrapper(
    const double & curvature, const geometry_msgs::msg::Twist & curr_speed,
    const double & pose_cost, const nav_msgs::msg::Path & path, double & linear_vel, double & sign)
  {
    return applyConstraints(
      curvature, curr_speed, pose_cost, path,
      linear_vel, sign);
  }

  double findVelocitySignChangeWrapper(
    const nav_msgs::msg::Path & transformed_plan)
  {
    return findVelocitySignChange(transformed_plan);
  }

  nav_msgs::msg::Path transformGlobalPlanWrapper(
    const geometry_msgs::msg::PoseStamped & pose)
  {
    return transformGlobalPlan(pose);
  }
};

TEST(RegulatedPurePursuitTest, basicAPI)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");

  // instantiate
  auto ctrl = std::make_shared<BasicAPIRPP>();
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();
  ctrl->deactivate();
  ctrl->cleanup();

  // setPlan and get plan
  nav_msgs::msg::Path path;
  path.poses.resize(2);
  path.poses[0].header.frame_id = "fake_frame";
  ctrl->setPlan(path);
  EXPECT_EQ(ctrl->getPlan().poses.size(), 2ul);
  EXPECT_EQ(ctrl->getPlan().poses[0].header.frame_id, std::string("fake_frame"));

  // set speed limit
  const double base_speed = ctrl->getSpeed();
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
  ctrl->setSpeedLimit(0.51, false);
  EXPECT_EQ(ctrl->getSpeed(), 0.51);
  ctrl->setSpeedLimit(nav2_costmap_2d::NO_SPEED_LIMIT, false);
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
  ctrl->setSpeedLimit(30, true);
  EXPECT_EQ(ctrl->getSpeed(), base_speed * 0.3);
  ctrl->setSpeedLimit(nav2_costmap_2d::NO_SPEED_LIMIT, true);
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
}

TEST(RegulatedPurePursuitTest, createCarrotMsg)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "Hi!";
  pose.pose.position.x = 1.0;
  pose.pose.position.y = 12.0;
  pose.pose.orientation.w = 0.5;

  auto rtn = ctrl->createCarrotMsgWrapper(pose);
  EXPECT_EQ(rtn->header.frame_id, std::string("Hi!"));
  EXPECT_EQ(rtn->point.x, 1.0);
  EXPECT_EQ(rtn->point.y, 12.0);
  EXPECT_EQ(rtn->point.z, 0.01);
}

TEST(RegulatedPurePursuitTest, findVelocitySignChange)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPPfindVelocitySignChange");
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "smb";
  auto time = node->get_clock()->now();
  pose.header.stamp = time;
  pose.pose.position.x = 1.0;
  pose.pose.position.y = 0.0;

  nav_msgs::msg::Path path;
  path.poses.resize(3);
  path.header.frame_id = "smb";
  path.header.stamp = pose.header.stamp;
  path.poses[0].pose.position.x = 1.0;
  path.poses[0].pose.position.y = 1.0;
  path.poses[1].pose.position.x = 2.0;
  path.poses[1].pose.position.y = 2.0;
  path.poses[2].pose.position.x = -1.0;
  path.poses[2].pose.position.y = -1.0;
  ctrl->setPlan(path);
  auto rtn = ctrl->findVelocitySignChangeWrapper(path);
  EXPECT_EQ(rtn, sqrt(8.0));

  path.poses[2].pose.position.x = 3.0;
  path.poses[2].pose.position.y = 3.0;
  ctrl->setPlan(path);
  rtn = ctrl->findVelocitySignChangeWrapper(path);
  EXPECT_EQ(rtn, std::numeric_limits<double>::max());
}

TEST(RegulatedPurePursuitTest, short_cusp_paths_and_rotation_duplicates)
{
  BasicAPIRPP controller;
  nav_msgs::msg::Path path;
  const auto no_cusp = std::numeric_limits<double>::max();
  EXPECT_EQ(controller.findVelocitySignChangeWrapper(path), no_cusp);
  geometry_msgs::msg::PoseStamped pose;
  pose.pose.position.x = 1.0;
  pose.pose.orientation.w = 1.0;
  path.poses.push_back(pose);
  EXPECT_EQ(controller.findVelocitySignChangeWrapper(path), no_cusp);
  path.poses.push_back(pose);
  EXPECT_EQ(controller.findVelocitySignChangeWrapper(path), no_cusp);

  // Same XY and yaw is an ordinary duplicate. Changed yaw is an in-place rotation cusp.
  path.poses.back().pose.orientation.z = std::sin(0.25);
  path.poses.back().pose.orientation.w = std::cos(0.25);
  EXPECT_DOUBLE_EQ(controller.findVelocitySignChangeWrapper(path), 1.0);
  path.poses.back().pose.orientation = pose.pose.orientation;
  path.poses.push_back(pose);
  path.poses.back().pose.position.x = 2.0;
  EXPECT_EQ(controller.findVelocitySignChangeWrapper(path), no_cusp);
  path.poses.front().pose.position.x = 0.0;
  path.poses.back().pose.position.x = 0.0;
  path.poses.insert(path.poses.begin() + 2, pose);
  EXPECT_DOUBLE_EQ(controller.findVelocitySignChangeWrapper(path), 1.0);
}

using CircleSegmentIntersectionParam = std::tuple<
  std::pair<double, double>,
  std::pair<double, double>,
  double,
  std::pair<double, double>
>;

class CircleSegmentIntersectionTest
  : public ::testing::TestWithParam<CircleSegmentIntersectionParam>
{};

TEST_P(CircleSegmentIntersectionTest, circleSegmentIntersection)
{
  auto pair1 = std::get<0>(GetParam());
  auto pair2 = std::get<1>(GetParam());
  auto r = std::get<2>(GetParam());
  auto expected_pair = std::get<3>(GetParam());
  auto pair_to_point = [](std::pair<double, double> p) -> geometry_msgs::msg::Point {
      geometry_msgs::msg::Point point;
      point.x = p.first;
      point.y = p.second;
      point.z = 0.0;
      return point;
    };
  auto p1 = pair_to_point(pair1);
  auto p2 = pair_to_point(pair2);
  auto actual = BasicAPIRPP::circleSegmentIntersectionWrapper(p1, p2, r);
  auto expected_point = pair_to_point(expected_pair);
  EXPECT_DOUBLE_EQ(actual.x, expected_point.x);
  EXPECT_DOUBLE_EQ(actual.y, expected_point.y);
  // Expect that the intersection point is actually r away from the origin
  EXPECT_DOUBLE_EQ(r, std::hypot(actual.x, actual.y));
}

INSTANTIATE_TEST_SUITE_P(
  InterpolationTest,
  CircleSegmentIntersectionTest,
  testing::Values(
    // Origin to the positive X axis
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {2.0, 0.0},
  1.0,
  {1.0, 0.0}
},
    // Origin to hte negative X axis
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {-2.0, 0.0},
  1.0,
  {-1.0, 0.0}
},
    // Origin to the positive Y axis
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {0.0, 2.0},
  1.0,
  {0.0, 1.0}
},
    // Origin to the negative Y axis
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {0.0, -2.0},
  1.0,
  {0.0, -1.0}
},
    // non-origin to the X axis with non-unit circle, with the second point inside
    CircleSegmentIntersectionParam{
  {4.0, 0.0},
  {-1.0, 0.0},
  2.0,
  {2.0, 0.0}
},
    // non-origin to the Y axis with non-unit circle, with the second point inside
    CircleSegmentIntersectionParam{
  {0.0, 4.0},
  {0.0, -0.5},
  2.0,
  {0.0, 2.0}
},
    // origin to the positive X axis, on the circle
    CircleSegmentIntersectionParam{
  {2.0, 0.0},
  {0.0, 0.0},
  2.0,
  {2.0, 0.0}
},
    // origin to the positive Y axis, on the circle
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {0.0, 2.0},
  2.0,
  {0.0, 2.0}
},
    // origin to the upper-right quadrant (3-4-5 triangle)
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {6.0, 8.0},
  5.0,
  {3.0, 4.0}
},
    // origin to the lower-left quadrant (3-4-5 triangle)
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {-6.0, -8.0},
  5.0,
  {-3.0, -4.0}
},
    // origin to the upper-left quadrant (3-4-5 triangle)
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {-6.0, 8.0},
  5.0,
  {-3.0, 4.0}
},
    // origin to the lower-right quadrant (3-4-5 triangle)
    CircleSegmentIntersectionParam{
  {0.0, 0.0},
  {6.0, -8.0},
  5.0,
  {3.0, -4.0}
}
));

TEST(RegulatedPurePursuitTest, lookaheadAPI)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);
  ctrl->configure(node, name, tf, costmap);

  geometry_msgs::msg::Twist twist;

  // test getLookAheadDistance
  double rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.6);  // default lookahead_dist

  // shouldn't be a function of speed
  twist.linear.x = 10.0;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.6);

  // now it should be a function of velocity, max out
  ctrl->setVelocityScaledLookAhead();
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.9);  // 10 speed maxes out at max_lookahead_dist

  // check normal range
  twist.linear.x = 0.35;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_NEAR(rtn, 0.525, 0.0001);  // 1.5 * 0.35

  // check minimum range
  twist.linear.x = 0.0;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.3);

  // test getLookAheadPoint
  double dist = 1.0;
  nav_msgs::msg::Path path;
  path.poses.resize(10);
  for (uint i = 0; i != path.poses.size(); i++) {
    path.poses[i].pose.position.x = static_cast<double>(i);
  }

  // test exact hits
  auto pt = ctrl->getLookAheadPointWrapper(dist, path);
  EXPECT_EQ(pt.pose.position.x, 1.0);

  // test getting next closest point without interpolation
  node->set_parameter(
    rclcpp::Parameter(
      name + ".use_interpolation",
      rclcpp::ParameterValue(false)));
  ctrl->configure(node, name, tf, costmap);
  dist = 3.8;
  pt = ctrl->getLookAheadPointWrapper(dist, path);
  EXPECT_EQ(pt.pose.position.x, 4.0);

  // test end of path
  dist = 100.0;
  pt = ctrl->getLookAheadPointWrapper(dist, path);
  EXPECT_EQ(pt.pose.position.x, 9.0);

  // test interpolation
  node->set_parameter(
    rclcpp::Parameter(
      name + ".use_interpolation",
      rclcpp::ParameterValue(true)));
  ctrl->configure(node, name, tf, costmap);
  dist = 3.8;
  pt = ctrl->getLookAheadPointWrapper(dist, path);
  EXPECT_EQ(pt.pose.position.x, 3.8);
}

TEST(RegulatedPurePursuitTest, rotateTests)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);
  ctrl->configure(node, name, tf, costmap);

  // shouldRotateToPath
  geometry_msgs::msg::PoseStamped carrot;
  double angle_to_path_rtn;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), false);

  carrot.pose.position.x = 0.5;
  carrot.pose.position.y = 0.25;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), false);

  carrot.pose.position.x = 0.5;
  carrot.pose.position.y = 1.0;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), true);

  // shouldRotateToGoalHeading
  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.0;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);

  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.24;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);

  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.26;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), false);

  // rotateToHeading
  double lin_v = 10.0;
  double ang_v = 0.5;
  double angle_to_path = 0.4;
  geometry_msgs::msg::Twist curr_speed;
  curr_speed.angular.z = 1.75;

  // basic full speed at a speed
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_EQ(lin_v, 0.0);
  EXPECT_EQ(ang_v, 1.6);  // Slow down before overshooting the remaining angle.

  // negative direction
  angle_to_path = -0.4;
  curr_speed.angular.z = -1.75;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_EQ(ang_v, -1.6);

  // kinematic clamping, no speed, some speed accelerating, some speed decelerating
  angle_to_path = 0.4;
  curr_speed.angular.z = 0.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 0.16, 0.01);

  curr_speed.angular.z = 1.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 1.16, 0.01);

  angle_to_path = -0.4;
  curr_speed.angular.z = 1.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 0.84, 0.01);
}

TEST(RegulatedPurePursuitTest, applyConstraints)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  constexpr double approach_velocity_scaling_dist = 0.6;
  nav2_util::declare_parameter_if_not_declared(
    node,
    name + ".approach_velocity_scaling_dist",
    rclcpp::ParameterValue(approach_velocity_scaling_dist));

  ctrl->configure(node, name, tf, costmap);

  auto no_approach_path = path_utils::generate_path(
    geometry_msgs::msg::PoseStamped(), 0.1, {
    std::make_unique<path_utils::Straight>(approach_velocity_scaling_dist + 1.0)
  });

  double curvature = 0.5;
  geometry_msgs::msg::Twist curr_speed;
  double pose_cost = 0.0;
  double linear_vel = 0.0;
  double sign = 1.0;

  // test curvature regulation (default)
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_EQ(linear_vel, 0.25);  // min set speed

  linear_vel = 1.0;
  curvature = 0.7407;
  curr_speed.linear.x = 0.5;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.5, 0.01);  // lower by curvature

  linear_vel = 1.0;
  curvature = 1000.0;
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.25, 0.01);  // min out by curvature

  // Approach velocity scaling on a path with no distance left
  auto approach_path = path_utils::generate_path(
    geometry_msgs::msg::PoseStamped(), 0.1, {
    std::make_unique<path_utils::Straight>(0.0)
  });

  linear_vel = 1.0;
  curvature = 0.0;
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.05, 0.01);  // min out on min approach velocity

  // now try with cost regulation (turn off velocity and only cost)
  // ctrl->setCostRegulationScaling();
  // ctrl->resetVelocityRegulationScaling();
  // curvature = 0.0;

  // min changable cost
  // pose_cost = 1;
  // linear_vel = 0.5;
  // curr_speed.linear.x = 0.5;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.498, 0.01);

  // max changing cost
  // pose_cost = 127;
  // curr_speed.linear.x = 0.255;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.255, 0.01);

  // over max cost thresh
  // pose_cost = 200;
  // curr_speed.linear.x = 0.25;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.25, 0.01);

  // test kinematic clamping
  // pose_cost = 200;
  // curr_speed.linear.x = 1.0;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.5, 0.01);
}

TEST(RegulatedPurePursuitTest, testDynamicParameter)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("Smactest");
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  auto ctrl =
    std::make_unique<nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController>();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  ctrl->configure(node, "test", tf, costmap);
  ctrl->activate();

  auto rec_param = std::make_shared<rclcpp::AsyncParametersClient>(
    node->get_node_base_interface(), node->get_node_topics_interface(),
    node->get_node_graph_interface(),
    node->get_node_services_interface());

  auto results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.desired_linear_vel", 1.0),
      rclcpp::Parameter("test.lookahead_dist", 7.0),
      rclcpp::Parameter("test.max_lookahead_dist", 7.0),
      rclcpp::Parameter("test.min_lookahead_dist", 6.0),
      rclcpp::Parameter("test.lookahead_time", 1.8),
      rclcpp::Parameter("test.rotate_to_heading_angular_vel", 18.0),
      rclcpp::Parameter("test.min_approach_linear_velocity", 1.0),
      rclcpp::Parameter("test.max_allowed_time_to_collision_up_to_carrot", 2.0),
      rclcpp::Parameter("test.cost_scaling_dist", 2.0),
      rclcpp::Parameter("test.cost_scaling_gain", 4.0),
      rclcpp::Parameter("test.regulated_linear_scaling_min_radius", 10.0),
      rclcpp::Parameter("test.transform_tolerance", 30.0),
      rclcpp::Parameter("test.max_angular_accel", 3.0),
      rclcpp::Parameter("test.rotate_to_heading_min_angle", 0.7),
      rclcpp::Parameter("test.regulated_linear_scaling_min_speed", 4.0),
      rclcpp::Parameter("test.use_velocity_scaled_lookahead_dist", false),
      rclcpp::Parameter("test.use_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test.use_cost_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test.allow_reversing", false),
      rclcpp::Parameter("test.use_rotate_to_heading", false)});

  rclcpp::spin_until_future_complete(
    node->get_node_base_interface(),
    results);

  EXPECT_EQ(node->get_parameter("test.desired_linear_vel").as_double(), 1.0);
  EXPECT_EQ(node->get_parameter("test.lookahead_dist").as_double(), 7.0);
  EXPECT_EQ(node->get_parameter("test.max_lookahead_dist").as_double(), 7.0);
  EXPECT_EQ(node->get_parameter("test.min_lookahead_dist").as_double(), 6.0);
  EXPECT_EQ(node->get_parameter("test.lookahead_time").as_double(), 1.8);
  EXPECT_EQ(node->get_parameter("test.rotate_to_heading_angular_vel").as_double(), 18.0);
  EXPECT_EQ(node->get_parameter("test.min_approach_linear_velocity").as_double(), 1.0);
  EXPECT_EQ(
    node->get_parameter(
      "test.max_allowed_time_to_collision_up_to_carrot").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.cost_scaling_dist").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.cost_scaling_gain").as_double(), 4.0);
  EXPECT_EQ(node->get_parameter("test.regulated_linear_scaling_min_radius").as_double(), 10.0);
  EXPECT_EQ(node->get_parameter("test.transform_tolerance").as_double(), 30.0);
  EXPECT_EQ(node->get_parameter("test.max_angular_accel").as_double(), 3.0);
  EXPECT_EQ(node->get_parameter("test.rotate_to_heading_min_angle").as_double(), 0.7);
  EXPECT_EQ(node->get_parameter("test.regulated_linear_scaling_min_speed").as_double(), 4.0);
  EXPECT_EQ(node->get_parameter("test.use_velocity_scaled_lookahead_dist").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.use_regulated_linear_velocity_scaling").as_bool(), false);
  EXPECT_EQ(
    node->get_parameter(
      "test.use_cost_regulated_linear_velocity_scaling").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.allow_reversing").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.use_rotate_to_heading").as_bool(), false);
}

class TransformGlobalPlanTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    ctrl_ = std::make_shared<BasicAPIRPP>();
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testRPP");
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  }

  void configure_costmap(uint16_t width, double resolution)
  {
    constexpr char costmap_frame[] = "test_costmap_frame";
    constexpr char robot_frame[] = "test_robot_frame";

    auto results = costmap_->set_parameters(
    {
      rclcpp::Parameter("global_frame", costmap_frame),
      rclcpp::Parameter("robot_base_frame", robot_frame),
      rclcpp::Parameter("width", width),
      rclcpp::Parameter("height", width),
      rclcpp::Parameter("resolution", resolution)
    });
    for (const auto & result : results) {
      EXPECT_TRUE(result.successful) << result.reason;
    }

    rclcpp_lifecycle::State state;
    costmap_->on_configure(state);
  }

  void configure_controller(double max_robot_pose_search_dist)
  {
    std::string plugin_name = "test_rpp";
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name + ".max_robot_pose_search_dist",
      rclcpp::ParameterValue(max_robot_pose_search_dist));
    ctrl_->configure(node_, plugin_name, tf_buffer_, costmap_);
  }

  void setup_transforms(geometry_msgs::msg::Point & robot_position)
  {
    transform_time_ = node_->get_clock()->now();
    // Note: transforms go parent to child

    // We will have a separate path and costmap frame for completeness,
    // but we will leave them cooincident for convenience.
    geometry_msgs::msg::TransformStamped path_to_costmap;
    path_to_costmap.header.frame_id = PATH_FRAME;
    path_to_costmap.header.stamp = transform_time_;
    path_to_costmap.child_frame_id = COSTMAP_FRAME;
    path_to_costmap.transform.translation.x = 0.0;
    path_to_costmap.transform.translation.y = 0.0;
    path_to_costmap.transform.translation.z = 0.0;

    geometry_msgs::msg::TransformStamped costmap_to_robot;
    costmap_to_robot.header.frame_id = COSTMAP_FRAME;
    costmap_to_robot.header.stamp = transform_time_;
    costmap_to_robot.child_frame_id = ROBOT_FRAME;
    costmap_to_robot.transform.translation.x = robot_position.x;
    costmap_to_robot.transform.translation.y = robot_position.y;
    costmap_to_robot.transform.translation.z = robot_position.z;

    tf2_msgs::msg::TFMessage tf_message;
    tf_message.transforms = {
      path_to_costmap,
      costmap_to_robot
    };
    for (const auto & transform : tf_message.transforms) {
      tf_buffer_->setTransform(transform, "test", false);
    }
    tf_buffer_->setUsingDedicatedThread(true);  // lying to let it do transforms
  }

  static constexpr char PATH_FRAME[] = "test_path_frame";
  static constexpr char COSTMAP_FRAME[] = "test_costmap_frame";
  static constexpr char ROBOT_FRAME[] = "test_robot_frame";

  std::shared_ptr<BasicAPIRPP> ctrl_;
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  rclcpp::Time transform_time_;
};

// This tests that not only should nothing get pruned on a costmap
// that contains the entire global_plan, and also that it doesn't skip to the end of the path
// which is closer to the robot pose than the start.
TEST_F(TransformGlobalPlanTest, no_pruning_on_large_costmap)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  robot_pose.pose.position.x = -0.1;
  robot_pose.pose.position.y = 0.0;
  robot_pose.pose.position.z = 0.0;
  // A really big costmap
  // the max_costmap_extent should be 50m
  configure_costmap(100u, 0.1);
  configure_controller(5.0);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  constexpr double spacing = 0.1;
  constexpr double circle_radius = 1.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::LeftCircle>(circle_radius)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan

  auto transformed_plan = ctrl_->transformGlobalPlanWrapper(robot_pose);
  EXPECT_EQ(transformed_plan.poses.size(), global_plan.poses.size());
}

// This plan shouldn't get pruned because of the costmap,
// but should be half pruned because it is halfway around the circle
TEST_F(TransformGlobalPlanTest, transform_start_selection)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  robot_pose.pose.position.x = 0.0;
  robot_pose.pose.position.y = 4.0;  // on the other side of the circle
  robot_pose.pose.position.z = 0.0;
  // Could set orientation going the other way, but RPP doesn't care
  constexpr double spacing = 0.1;
  constexpr double circle_radius = 2.0;  // diameter 4

  // A really big costmap
  // the max_costmap_extent should be 50m
  configure_costmap(100u, 0.1);
  // This should just be at least half the circumference: pi*r ~= 6
  constexpr double max_robot_pose_search_dist = 10.0;
  configure_controller(max_robot_pose_search_dist);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::LeftCircle>(circle_radius)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan
  auto transformed_plan = ctrl_->transformGlobalPlanWrapper(robot_pose);
  EXPECT_NEAR(transformed_plan.poses.size(), global_plan.poses.size() / 2, 1);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.x, 0.0, 0.5);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.y, 0.0, 0.5);
}

// This should throw an exception when all poses are outside of the costmap
TEST_F(TransformGlobalPlanTest, all_poses_outside_of_costmap)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  // far away from the path
  robot_pose.pose.position.x = 1000.0;
  robot_pose.pose.position.y = 1000.0;
  robot_pose.pose.position.z = 0.0;
  // Could set orientation going the other way, but RPP doesn't care
  constexpr double spacing = 0.1;
  constexpr double circle_radius = 2.0;  // diameter 4

  // A "normal" costmap
  // the max_costmap_extent should be 50m
  configure_costmap(10u, 0.1);
  // This should just be at least half the circumference: pi*r ~= 6
  constexpr double max_robot_pose_search_dist = 10.0;
  configure_controller(max_robot_pose_search_dist);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::LeftCircle>(circle_radius)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan
  EXPECT_THROW(ctrl_->transformGlobalPlanWrapper(robot_pose), nav2_core::PlannerException);
}

// Should shortcut the circle if the circle is shorter than max_robot_pose_search_dist
TEST_F(TransformGlobalPlanTest, good_circle_shortcut)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  // far away from the path
  robot_pose.pose.position.x = -0.1;
  robot_pose.pose.position.y = 0.0;
  robot_pose.pose.position.z = 0.0;
  // Could set orientation going the other way, but RPP doesn't care
  constexpr double spacing = 0.1;
  constexpr double circle_radius = 2.0;  // diameter 4

  // A "normal" costmap
  // the max_costmap_extent should be 50m
  configure_costmap(100u, 0.1);
  // This should just be at least the circumference: 2*pi*r ~= 12
  constexpr double max_robot_pose_search_dist = 15.0;
  configure_controller(max_robot_pose_search_dist);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::LeftCircle>(circle_radius)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan
  auto transformed_plan = ctrl_->transformGlobalPlanWrapper(robot_pose);
  EXPECT_NEAR(transformed_plan.poses.size(), 1, 1);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.x, 0.0, 0.5);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.y, 0.0, 0.5);
}

// Simple costmap pruning on a straight line
TEST_F(TransformGlobalPlanTest, costmap_pruning)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  // far away from the path
  robot_pose.pose.position.x = -0.1;
  robot_pose.pose.position.y = 0.0;
  robot_pose.pose.position.z = 0.0;
  // Could set orientation going the other way, but RPP doesn't care
  constexpr double spacing = 1.0;

  // A "normal" costmap
  // the max_costmap_extent should be 50m
  configure_costmap(20u, 0.5);
  constexpr double max_robot_pose_search_dist = 10.0;
  configure_controller(max_robot_pose_search_dist);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  constexpr double path_length = 100.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::Straight>(path_length)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan
  auto transformed_plan = ctrl_->transformGlobalPlanWrapper(robot_pose);
  EXPECT_NEAR(transformed_plan.poses.size(), 10u, 1);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.x, 0.0, 0.5);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.y, 0.0, 0.5);
}

// Should prune out later portions of the path that come back into the costmap
TEST_F(TransformGlobalPlanTest, prune_after_leaving_costmap)
{
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = COSTMAP_FRAME;
  robot_pose.header.stamp = transform_time_;
  // far away from the path
  robot_pose.pose.position.x = -0.1;
  robot_pose.pose.position.y = 0.0;
  robot_pose.pose.position.z = 0.0;
  // Could set orientation going the other way, but RPP doesn't care
  constexpr double spacing = 1.0;

  // A "normal" costmap
  // the max_costmap_extent should be 50m
  configure_costmap(20u, 0.5);
  constexpr double max_robot_pose_search_dist = 10.0;
  configure_controller(max_robot_pose_search_dist);
  setup_transforms(robot_pose.pose.position);

  // Set up test path;

  geometry_msgs::msg::PoseStamped start_of_path;
  start_of_path.header.frame_id = PATH_FRAME;
  start_of_path.header.stamp = transform_time_;
  start_of_path.pose.position.x = 0.0;
  start_of_path.pose.position.y = 0.0;
  start_of_path.pose.position.z = 0.0;

  constexpr double path_length = 100.0;

  auto global_plan = path_utils::generate_path(
    start_of_path, spacing, {
    std::make_unique<path_utils::Straight>(path_length),
    std::make_unique<path_utils::LeftTurnAround>(1.0),
    std::make_unique<path_utils::Straight>(path_length)
  });

  ctrl_->setPlan(global_plan);

  // Transform the plan
  auto transformed_plan = ctrl_->transformGlobalPlanWrapper(robot_pose);
  // This should be essentially the same as the regular straight path
  EXPECT_NEAR(transformed_plan.poses.size(), 10u, 1);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.x, 0.0, 0.5);
  EXPECT_NEAR(transformed_plan.poses[0].pose.position.y, 0.0, 0.5);
}

class AngularLimitGoalChecker : public nav2_core::GoalChecker
{
public:
  void initialize(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr &, const std::string &,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override {}
  void reset() override {}
  bool isGoalReached(
    const geometry_msgs::msg::Pose &, const geometry_msgs::msg::Pose &,
    const geometry_msgs::msg::Twist &) override {return false;}
  bool getTolerances(
    geometry_msgs::msg::Pose & pose_tolerance, geometry_msgs::msg::Twist &) override
  {
    pose_tolerance.position.x = tolerance;
    return true;
  }

  double tolerance{0.1};
};

class CostmapLockGoalChecker : public AngularLimitGoalChecker
{
public:
  explicit CostmapLockGoalChecker(nav2_costmap_2d::Costmap2D::mutex_t & mutex)
  : mutex_(mutex) {}

  bool getTolerances(
    geometry_msgs::msg::Pose & pose_tolerance, geometry_msgs::msg::Twist & twist_tolerance) override
  {
    // A different thread must fail to acquire the map while the controller is
    // evaluating it. Trying from this thread would always pass for a recursive mutex.
    bool acquired = false;
    std::thread observer([&]() {
        acquired = mutex_.try_lock();
        if (acquired) {
          mutex_.unlock();
        }
      });
    observer.join();
    locked_during_evaluation.push_back(!acquired);
    return AngularLimitGoalChecker::getTolerances(pose_tolerance, twist_tolerance);
  }

  std::vector<bool> locked_during_evaluation;

private:
  nav2_costmap_2d::Costmap2D::mutex_t & mutex_;
};

class AngularVelocityLimitTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("angular_limit_test");
    tf_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>("angular_limit_costmap");
    const auto results = costmap_->set_parameters(
    {
      rclcpp::Parameter("plugins", std::vector<std::string>{}),
      rclcpp::Parameter("origin_x", -5.0),
      rclcpp::Parameter("origin_y", -5.0),
      rclcpp::Parameter("width", 10),
      rclcpp::Parameter("height", 10),
      rclcpp::Parameter("resolution", 0.05)});
    for (const auto & result : results) {
      ASSERT_TRUE(result.successful) << result.reason;
    }
    costmap_->on_configure(rclcpp_lifecycle::State());

    for (const auto & parameter : std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("test_rpp.desired_linear_vel", 1.0),
      rclcpp::Parameter("test_rpp.lookahead_dist", 0.6),
      rclcpp::Parameter("test_rpp.transform_tolerance", 0.0),
      rclcpp::Parameter("test_rpp.use_interpolation", false),
      rclcpp::Parameter("test_rpp.use_velocity_scaled_lookahead_dist", false),
      rclcpp::Parameter("test_rpp.use_rotate_to_heading", false),
      rclcpp::Parameter("test_rpp.use_collision_detection", false),
      rclcpp::Parameter("test_rpp.use_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test_rpp.use_cost_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test_rpp.regulated_linear_scaling_min_speed", 0.1)})
    {
      node_->declare_parameter(parameter.get_name(), parameter.get_parameter_value());
    }

    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = costmap_->getGlobalFrameID();
    transform.child_frame_id = costmap_->getBaseFrameID();
    transform.transform.rotation.w = 1.0;
    ASSERT_TRUE(tf_->setTransform(transform, "angular_limit_test", true));
    ctrl_ = std::make_shared<BasicAPIRPP>();
    ctrl_->configure(node_, "test_rpp", tf_, costmap_);
    ctrl_->activate();
  }

  void TearDown() override
  {
    if (ctrl_) {
      ctrl_->deactivate();
      ctrl_->cleanup();
    }
  }

  void update(const std::vector<rclcpp::Parameter> & parameters)
  {
    const auto result = node_->set_parameters_atomically(parameters);
    ASSERT_TRUE(result.successful) << result.reason;
  }

  geometry_msgs::msg::Twist command(
    double carrot_x = 0.5, double carrot_y = 0.5, double current_angular_vel = 0.0)
  {
    geometry_msgs::msg::PoseStamped robot_pose;
    robot_pose.header.frame_id = costmap_->getGlobalFrameID();
    robot_pose.header.stamp = node_->now();
    robot_pose.pose.orientation.w = 1.0;
    nav_msgs::msg::Path path;
    path.header = robot_pose.header;
    path.poses.assign(3, robot_pose);
    path.poses[1].pose.position.x = carrot_x;
    path.poses[1].pose.position.y = carrot_y;
    path.poses[2].pose.position.x = 3.0 * carrot_x;
    path.poses[2].pose.position.y = 3.0 * carrot_y;
    ctrl_->setPlan(path);
    geometry_msgs::msg::Twist current_speed;
    current_speed.angular.z = current_angular_vel;
    return ctrl_->computeVelocityCommands(robot_pose, current_speed, &goal_checker_).twist;
  }

  std::shared_ptr<BasicAPIRPP> ctrl_;
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  AngularLimitGoalChecker goal_checker_;
};

TEST_F(AngularVelocityLimitTest, disabled_and_under_limit_preserve_commands)
{
  EXPECT_DOUBLE_EQ(node_->get_parameter("test_rpp.max_angular_vel").as_double(), -1.0);
  auto twist = command();  // The carrot at (0.5, 0.5) has curvature 2 / m.
  EXPECT_DOUBLE_EQ(twist.linear.x, 1.0);
  EXPECT_DOUBLE_EQ(twist.angular.z, 2.0);

  update({rclcpp::Parameter("test_rpp.max_angular_vel", 0.5)});
  twist = command(1.0, 0.1);
  EXPECT_DOUBLE_EQ(twist.linear.x, 1.0);
  EXPECT_NEAR(twist.angular.z, 0.2 / 1.01, 1e-12);
  twist = command(1.0, 0.0);
  EXPECT_DOUBLE_EQ(twist.linear.x, 1.0);
  EXPECT_DOUBLE_EQ(twist.angular.z, 0.0);
}

TEST_F(AngularVelocityLimitTest, preserves_curvature_for_both_speeds_and_directions)
{
  update(
  {
    rclcpp::Parameter("test_rpp.max_angular_vel", 0.5),
    rclcpp::Parameter("test_rpp.allow_reversing", true)});
  for (double desired_speed : {0.5, 1.0}) {
    update({rclcpp::Parameter("test_rpp.desired_linear_vel", desired_speed)});
    for (double direction : {-1.0, 1.0}) {
      for (double turn : {-1.0, 1.0}) {
        const auto twist = command(direction * 0.5, turn * 0.5);
        EXPECT_DOUBLE_EQ(twist.linear.x, direction * 0.25);
        EXPECT_DOUBLE_EQ(twist.angular.z, direction * turn * 0.5);
        EXPECT_DOUBLE_EQ(twist.angular.z / twist.linear.x, turn * 2.0);
      }
    }
  }
}

TEST_F(AngularVelocityLimitTest, angular_limit_takes_priority_over_minimum_linear_speed)
{
  update(
  {
    rclcpp::Parameter("test_rpp.use_regulated_linear_velocity_scaling", true),
    rclcpp::Parameter("test_rpp.regulated_linear_scaling_min_radius", 100.0)});
  auto twist = command();
  EXPECT_NEAR(twist.linear.x, 0.1, 1e-12);
  update({rclcpp::Parameter("test_rpp.max_angular_vel", 0.1)});
  twist = command();
  EXPECT_NEAR(twist.linear.x, 0.05, 1e-12);
  EXPECT_DOUBLE_EQ(twist.angular.z, 0.1);
}

TEST_F(AngularVelocityLimitTest, limits_rotation_even_when_measured_speed_exceeds_limit)
{
  update(
  {
    rclcpp::Parameter("test_rpp.max_angular_vel", 0.5),
    rclcpp::Parameter("test_rpp.use_rotate_to_heading", true)});
  for (double direction : {-1.0, 1.0}) {
    auto twist = command(0.1, direction, 0.0);
    EXPECT_DOUBLE_EQ(twist.linear.x, 0.0);
    EXPECT_NEAR(twist.angular.z, direction * 0.16, 1e-12);
    twist = command(0.1, direction, direction * 1.7);
    EXPECT_DOUBLE_EQ(twist.linear.x, 0.0);
    EXPECT_DOUBLE_EQ(twist.angular.z, direction * 0.5);
  }
}

TEST_F(AngularVelocityLimitTest, applies_heading_braking_after_acceleration_clamping)
{
  update({rclcpp::Parameter("test_rpp.max_angular_accel", 2.0)});
  for (double direction : {-1.0, 1.0}) {
    SCOPED_TRACE(direction);
    geometry_msgs::msg::Twist speed;
    speed.angular.z = direction * 0.3;
    double linear_vel = 1.0;
    double angular_vel = 0.0;

    // At 2 rad/s^2, 0.25 rad/s takes exactly 1/64 rad to stop. The
    // acceleration-limited command alone would increase the speed to 0.4.
    ctrl_->rotateToHeadingWrapper(
      linear_vel, angular_vel, direction / 64.0, speed);
    EXPECT_DOUBLE_EQ(linear_vel, 0.0);
    EXPECT_DOUBLE_EQ(angular_vel, direction * 0.25);

    // Starting from rest still respects the 0.05 s control period: the
    // first command is 0.1 rad/s even when the stopping bound is higher.
    speed.angular.z = 0.0;
    ctrl_->rotateToHeadingWrapper(
      linear_vel, angular_vel, direction / 64.0, speed);
    EXPECT_DOUBLE_EQ(angular_vel, direction * 0.1);

    // If already rotating away from the target, first decelerate in the
    // current direction rather than reversing the angular command immediately.
    speed.angular.z = -direction * 0.3;
    ctrl_->rotateToHeadingWrapper(
      linear_vel, angular_vel, direction / 64.0, speed);
    EXPECT_NEAR(angular_vel, -direction * 0.2, 1e-12);
  }

  double linear_vel = 1.0;
  double angular_vel = 0.0;
  geometry_msgs::msg::Twist speed;
  speed.angular.z = 0.3;
  ctrl_->rotateToHeadingWrapper(linear_vel, angular_vel, 0.0, speed);
  EXPECT_DOUBLE_EQ(linear_vel, 0.0);
  EXPECT_DOUBLE_EQ(angular_vel, 0.0);
}

TEST_F(AngularVelocityLimitTest, collision_check_uses_limited_commands)
{
  // The existing collision-detection switch is read during configuration only.
  ctrl_->deactivate();
  ctrl_->cleanup();
  update({rclcpp::Parameter("test_rpp.use_collision_detection", true)});
  ctrl_->configure(node_, "test_rpp", tf_, costmap_);
  ctrl_->activate();
  auto * map = costmap_->getCostmap();
  unsigned int wall_x, wall_y;
  ASSERT_TRUE(map->worldToMap(0.4, 0.0, wall_x, wall_y));
  for (unsigned int y = 0; y < map->getSizeInCellsY(); ++y) {
    map->setCost(wall_x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  EXPECT_THROW(command(), nav2_core::PlannerException);
  update({rclcpp::Parameter("test_rpp.max_angular_vel", 0.2)});
  EXPECT_NO_THROW(
  {
    const auto twist = command();
    EXPECT_NEAR(twist.linear.x, 0.1, 1e-12);
    EXPECT_DOUBLE_EQ(twist.angular.z, 0.2);
  });
}

TEST_F(AngularVelocityLimitTest, dynamic_updates_change_effective_limit)
{
  update({rclcpp::Parameter("test_rpp.max_angular_vel", 1.0)});
  EXPECT_DOUBLE_EQ(command().angular.z, 1.0);
  update({rclcpp::Parameter("test_rpp.max_angular_vel", 0.25)});
  const auto twist = command();
  EXPECT_DOUBLE_EQ(twist.linear.x, 0.125);
  EXPECT_DOUBLE_EQ(twist.angular.z, 0.25);
  update({rclcpp::Parameter("test_rpp.max_angular_vel", -1.0)});
  EXPECT_DOUBLE_EQ(command().angular.z, 2.0);
}

TEST_F(AngularVelocityLimitTest, rejects_invalid_updates_before_mutating_other_parameters)
{
  update({rclcpp::Parameter("test_rpp.max_angular_vel", 0.5)});
  for (double invalid : {0.0, -0.5, -2.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
  {
    const auto result = node_->set_parameters_atomically(
    {
      rclcpp::Parameter("test_rpp.desired_linear_vel", 0.125),
      rclcpp::Parameter("test_rpp.max_angular_vel", invalid)});
    EXPECT_FALSE(result.successful);
    EXPECT_FALSE(result.reason.empty());
    EXPECT_DOUBLE_EQ(node_->get_parameter("test_rpp.desired_linear_vel").as_double(), 1.0);
    EXPECT_DOUBLE_EQ(node_->get_parameter("test_rpp.max_angular_vel").as_double(), 0.5);
    EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 1.0);
    EXPECT_DOUBLE_EQ(command().angular.z, 0.5);
  }
  // Exercise the callback's type validation independently of ROS parameter type checks.
  const auto result = ctrl_->updateParameters(
  {
    rclcpp::Parameter("test_rpp.desired_linear_vel", 0.125),
    rclcpp::Parameter("test_rpp.max_angular_vel", 1)});
  EXPECT_FALSE(result.successful);
  EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 1.0);
  EXPECT_DOUBLE_EQ(command().angular.z, 0.5);
}

TEST_F(AngularVelocityLimitTest, rejects_invalid_initial_limits)
{
  for (double invalid : {0.0, -0.5, -2.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
  {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("invalid_angular_limit");
    node->declare_parameter("test_rpp.max_angular_vel", invalid);
    BasicAPIRPP controller;
    EXPECT_THROW(controller.configure(node, "test_rpp", tf_, costmap_), std::invalid_argument);
  }
}

class CandidateEvaluationTest : public AngularVelocityLimitTest
{
protected:
  nav_msgs::msg::Path candidatePlan(double carrot_y = 0.5)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = costmap_->getGlobalFrameID();
    path.header.stamp = node_->now();
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.orientation.w = 1.0;
    path.poses.assign(4, pose);
    path.poses[0].pose.position.x = -0.3;  // Would be pruned by the regular entry point.
    path.poses[2].pose.position.x = 0.5;
    path.poses[2].pose.position.y = carrot_y;
    path.poses[3].pose.position.x = 1.5;
    path.poses[3].pose.position.y = 3.0 * carrot_y;
    return path;
  }
};

class PathInversionTest : public AngularVelocityLimitTest
{
protected:
  void SetUp() override
  {
    AngularVelocityLimitTest::SetUp();
    // Declaring these explicitly also lets this test source run against the old implementation.
    for (const auto & parameter : std::vector<rclcpp::Parameter>{
        rclcpp::Parameter("test_rpp.enforce_path_inversion", false),
        rclcpp::Parameter("test_rpp.inversion_xy_tolerance", 0.2),
        rclcpp::Parameter("test_rpp.inversion_yaw_tolerance", 0.4)})
    {
      if (!node_->has_parameter(parameter.get_name())) {
        node_->declare_parameter(parameter.get_name(), parameter.get_parameter_value());
      }
    }
  }

  void enforceInversions()
  {
    update({rclcpp::Parameter("test_rpp.allow_reversing", true),
      rclcpp::Parameter("test_rpp.enforce_path_inversion", true)});
  }

  geometry_msgs::msg::PoseStamped robotPose(double x = 0.0, double y = 0.0, double yaw = 0.0)
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = costmap_->getGlobalFrameID();
    pose.header.stamp = node_->now();
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.z = std::sin(yaw / 2.0);
    pose.pose.orientation.w = std::cos(yaw / 2.0);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = costmap_->getGlobalFrameID();
    transform.child_frame_id = costmap_->getBaseFrameID();
    transform.transform.translation.x = x;
    transform.transform.translation.y = y;
    transform.transform.rotation = pose.pose.orientation;
    EXPECT_TRUE(tf_->setTransform(transform, "path_inversion_test", true));
    return pose;
  }

  nav_msgs::msg::Path pathThrough(std::initializer_list<std::pair<double, double>> points)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = costmap_->getGlobalFrameID();
    path.header.stamp = node_->now();
    for (const auto & point : points) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = point.first;
      pose.pose.position.y = point.second;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    return path;
  }
};

TEST_F(PathInversionTest, normal_yaml_flags_preserve_golden_commands)
{
  ctrl_->deactivate();
  ctrl_->cleanup();
  // RPP settings from common/planner/pure_pursuit.yaml; wrapper-only settings are unrelated.
  update({rclcpp::Parameter("test_rpp.lookahead_dist", 1.0),
    rclcpp::Parameter("test_rpp.min_lookahead_dist", 0.3),
    rclcpp::Parameter("test_rpp.max_lookahead_dist", 1.5),
    rclcpp::Parameter("test_rpp.lookahead_time", 1.5),
    rclcpp::Parameter("test_rpp.use_velocity_scaled_lookahead_dist", true),
    rclcpp::Parameter("test_rpp.use_interpolation", true),
    rclcpp::Parameter("test_rpp.use_rotate_to_heading", false),
    rclcpp::Parameter("test_rpp.allow_reversing", false),
    rclcpp::Parameter("test_rpp.use_collision_detection", true),
    rclcpp::Parameter("test_rpp.use_regulated_linear_velocity_scaling", true),
    rclcpp::Parameter("test_rpp.use_cost_regulated_linear_velocity_scaling", true),
    rclcpp::Parameter("test_rpp.cost_scaling_dist", 0.6),
    rclcpp::Parameter("test_rpp.inflation_cost_scaling_factor", 3.0),
    rclcpp::Parameter("test_rpp.regulated_linear_scaling_min_radius", 0.9),
    rclcpp::Parameter("test_rpp.max_angular_vel", 0.436332)});
  ctrl_->configure(node_, "test_rpp", tf_, costmap_);
  ctrl_->activate();
  EXPECT_FALSE(node_->get_parameter("test_rpp.enforce_path_inversion").as_bool());

  const auto straight = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}});
  const auto corner = pathThrough({{0.0, 0.0}, {0.5, 0.5}, {1.5, 1.5}});
  const auto overlapping = pathThrough(
    {{0.0, 0.0}, {0.0, 0.0}, {0.5, 0.5}, {1.5, 1.5}, {0.001, 0.002}});
  std::vector<nav_msgs::msg::Path> paths{straight, corner, overlapping, overlapping};
  const double corner_speed = 0.436332 * 0.3 / std::sqrt(2.0);
  const std::vector<double> expected_linear{1.0, corner_speed, corner_speed, 0.05};
  const std::vector<double> expected_angular{0.0, 0.436332, 0.436332, 0.0};
  for (size_t i = 0; i < paths.size(); ++i) {
    auto pose = i == 3 ? robotPose(0.001, 0.001) : robotPose();
    const auto implicit_off = ctrl_->evaluateVelocityCommands(
      paths[i], 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_);
    update({rclcpp::Parameter("test_rpp.enforce_path_inversion", false)});
    const auto explicit_off = ctrl_->evaluateVelocityCommands(
      paths[i], 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_);
    EXPECT_EQ(implicit_off.command, explicit_off.command);
    EXPECT_EQ(implicit_off.transformed_plan, explicit_off.transformed_plan);
    EXPECT_NEAR(implicit_off.command.twist.linear.x, expected_linear[i], 1e-12);
    EXPECT_NEAR(implicit_off.command.twist.angular.z, expected_angular[i], 1e-12);
    ctrl_->setPlan(paths[i]);
    EXPECT_EQ(ctrl_->computeVelocityCommands(
        pose, geometry_msgs::msg::Twist(), &goal_checker_), implicit_off.command);
  }
}

TEST_F(PathInversionTest, closer_future_reverse_segment_is_excluded)
{
  enforceInversions();
  const auto path = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.1}, {0.2, 0.1}});
  const auto pose = robotPose(0.2, 0.1);
  const auto result = ctrl_->evaluateVelocityCommands(
    path, 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_);
  ASSERT_EQ(result.transformed_plan.poses.size(), 3u);
  EXPECT_NEAR(result.transformed_plan.poses.front().pose.position.x, -0.2, 1e-12);
  EXPECT_NEAR(result.transformed_plan.poses.back().pose.position.x, 1.8, 1e-12);
  EXPECT_GT(result.command.twist.linear.x, 0.0);
}

TEST_F(PathInversionTest, inversion_requires_both_tolerances_and_set_plan_resets_progress)
{
  enforceInversions();
  const auto path = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  ctrl_->setPlan(path);
  auto pose = robotPose(1.7);
  EXPECT_GT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  pose = robotPose(1.9, 0.0, 0.5);
  EXPECT_GT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  pose = robotPose(1.9);
  EXPECT_LT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  // Moving beyond the tolerance does not re-enter the completed forward segment.
  pose = robotPose(0.8);
  EXPECT_LT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  ctrl_->setPlan(path);
  EXPECT_GT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
}

TEST_F(PathInversionTest, enforcement_requires_reversing_during_configuration)
{
  for (bool rotate : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("invalid_inversion_mode");
    node->declare_parameter("test_rpp.enforce_path_inversion", true);
    node->declare_parameter("test_rpp.use_rotate_to_heading", rotate);
    // The true/true case must be rejected after configure's existing reversal resolution.
    node->declare_parameter("test_rpp.allow_reversing", rotate);
    BasicAPIRPP controller;
    EXPECT_THROW(controller.configure(node, "test_rpp", tf_, costmap_), std::invalid_argument);
  }
}

TEST_F(PathInversionTest, enforcement_flag_batches_validate_before_mutating_state)
{
  const auto rejected_enable = node_->set_parameters_atomically(
    {rclcpp::Parameter("test_rpp.enforce_path_inversion", true)});
  EXPECT_FALSE(rejected_enable.successful);
  EXPECT_FALSE(node_->get_parameter("test_rpp.enforce_path_inversion").as_bool());
  for (bool reverse_order : {false, true}) {
    std::vector<rclcpp::Parameter> enable{
      rclcpp::Parameter("test_rpp.allow_reversing", true),
      rclcpp::Parameter("test_rpp.enforce_path_inversion", true)};
    if (reverse_order) {
      std::reverse(enable.begin(), enable.end());
    }
    update(enable);
    const auto rejected_disable = node_->set_parameters_atomically(
      {rclcpp::Parameter("test_rpp.desired_linear_vel", 0.125),
        rclcpp::Parameter("test_rpp.allow_reversing", false)});
    EXPECT_FALSE(rejected_disable.successful);
    EXPECT_TRUE(node_->get_parameter("test_rpp.enforce_path_inversion").as_bool());
    EXPECT_TRUE(node_->get_parameter("test_rpp.allow_reversing").as_bool());
    EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 1.0);
    const auto rejected_rotation = node_->set_parameters_atomically(
      {rclcpp::Parameter("test_rpp.use_rotate_to_heading", true)});
    EXPECT_FALSE(rejected_rotation.successful);
    EXPECT_FALSE(node_->get_parameter("test_rpp.use_rotate_to_heading").as_bool());
    std::vector<rclcpp::Parameter> disable{
      rclcpp::Parameter("test_rpp.allow_reversing", false),
      rclcpp::Parameter("test_rpp.enforce_path_inversion", false)};
    if (reverse_order) {
      std::reverse(disable.begin(), disable.end());
    }
    update(disable);
  }
}

TEST_F(PathInversionTest, unsteerable_cusp_heading_fails_without_advancing_progress)
{
  enforceInversions();
  const auto path = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  for (double x : {2.0, 1.99, 2.01}) {
    ctrl_->setPlan(path);
    auto pose = robotPose(x, 0.0, 0.5);
    EXPECT_THROW(ctrl_->computeVelocityCommands(
        pose, geometry_msgs::msg::Twist(), &goal_checker_), nav2_core::PlannerException);
    EXPECT_THROW(ctrl_->evaluateVelocityCommands(
        path, 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_), nav2_core::PlannerException);
    // Failed adopted evaluations must not release the reverse segment.
    pose = robotPose(0.8);
    EXPECT_GT(ctrl_->computeVelocityCommands(
        pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
    // Once heading is corrected at the cusp, the same plan can proceed in reverse.
    pose = robotPose(x);
    EXPECT_LT(ctrl_->computeVelocityCommands(
        pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  }
}

TEST_F(PathInversionTest, adopted_progress_crosses_multiple_inversions_in_order)
{
  enforceInversions();
  const auto path = pathThrough(
    {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}});
  ctrl_->setPlan(path);
  EXPECT_LT(ctrl_->computeVelocityCommands(
      robotPose(1.9), geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  EXPECT_GT(ctrl_->computeVelocityCommands(
      robotPose(0.1), geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  EXPECT_GT(ctrl_->computeVelocityCommands(
      robotPose(0.5), geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
}

TEST_F(PathInversionTest, heading_guard_checks_the_next_pending_cusp)
{
  enforceInversions();
  auto path = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {1.02, 0.0}, {1.0, 0.0}, {1.5, 0.0}});
  path.poses[3].pose.orientation.z = std::sin(0.25);
  path.poses[3].pose.orientation.w = std::cos(0.25);
  ctrl_->setPlan(path);
  EXPECT_THROW(ctrl_->computeVelocityCommands(
      robotPose(1.02), geometry_msgs::msg::Twist(), &goal_checker_), nav2_core::PlannerException);
}

TEST_F(PathInversionTest, duplicate_positions_do_not_hide_a_reversal)
{
  enforceInversions();
  const auto path = pathThrough(
    {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {-0.5, 0.0}});
  const auto result = ctrl_->evaluateVelocityCommands(
    path, 1.0, robotPose(), geometry_msgs::msg::Twist(), &goal_checker_);
  ASSERT_EQ(result.transformed_plan.poses.size(), 4u);
  EXPECT_DOUBLE_EQ(result.transformed_plan.poses.back().pose.position.x, 1.0);
  EXPECT_GT(result.command.twist.linear.x, 0.0);
}

TEST_F(PathInversionTest, rotation_tie_handling_is_confined_to_reversing_paths)
{
  auto path = pathThrough({{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});
  for (size_t i : {2u, 3u}) {
    path.poses[i].pose.orientation.z = std::sin(M_PI / 4.0);
    path.poses[i].pose.orientation.w = std::cos(M_PI / 4.0);
  }
  const auto pose = robotPose(1.0, 0.0, M_PI / 2.0);
  update({rclcpp::Parameter("test_rpp.allow_reversing", true)});
  const auto reversing = ctrl_->evaluateVelocityCommands(
    path, 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_);
  ASSERT_EQ(reversing.transformed_plan.poses.size(), 2u);
  EXPECT_NEAR(tf2::getYaw(reversing.transformed_plan.poses.front().pose.orientation), 0.0, 1e-12);
  update({rclcpp::Parameter("test_rpp.allow_reversing", false)});
  const auto normal = ctrl_->evaluateVelocityCommands(
    path, 1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_);
  ASSERT_EQ(normal.transformed_plan.poses.size(), 3u);
  EXPECT_NEAR(tf2::getYaw(normal.transformed_plan.poses.front().pose.orientation), -M_PI / 2.0, 1e-12);
}

TEST_F(PathInversionTest, candidates_and_failures_leave_adopted_progress_unchanged)
{
  enforceInversions();
  const auto adopted = pathThrough(
    {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  ctrl_->setPlan(adopted);
  auto pose = robotPose(1.9);
  ASSERT_LT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
  const auto candidate = pathThrough(
    {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.1}, {0.2, 0.1}});
  pose = robotPose(0.2, 0.1);
  const auto first = ctrl_->evaluateVelocityCommands(
    candidate, 0.3, pose, geometry_msgs::msg::Twist(), &goal_checker_);
  ASSERT_EQ(first.transformed_plan.poses.size(), 3u);
  ctrl_->evaluateVelocityCommands(
    adopted, 0.1, robotPose(1.9), geometry_msgs::msg::Twist(), &goal_checker_);
  EXPECT_THROW(ctrl_->evaluateVelocityCommands(
      nav_msgs::msg::Path(), 0.3, pose, geometry_msgs::msg::Twist(), &goal_checker_),
    nav2_core::PlannerException);
  EXPECT_THROW(ctrl_->evaluateVelocityCommands(
      candidate, -1.0, pose, geometry_msgs::msg::Twist(), &goal_checker_),
    nav2_core::PlannerException);
  EXPECT_THROW(ctrl_->evaluateVelocityCommands(
      adopted, 0.3, robotPose(2.0, 0.0, 0.5), geometry_msgs::msg::Twist(), &goal_checker_),
    nav2_core::PlannerException);
  pose = robotPose(0.2, 0.1);
  const auto second = ctrl_->evaluateVelocityCommands(
    candidate, 0.3, pose, geometry_msgs::msg::Twist(), &goal_checker_);
  // Header timestamps can differ, so compare the actual command and path geometry.
  EXPECT_EQ(first.command.twist, second.command.twist);
  ASSERT_EQ(first.transformed_plan.poses.size(), second.transformed_plan.poses.size());
  for (size_t i = 0; i < first.transformed_plan.poses.size(); ++i) {
    EXPECT_EQ(first.transformed_plan.poses[i].pose, second.transformed_plan.poses[i].pose);
  }
  EXPECT_EQ(ctrl_->getPlan(), adopted);
  EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 1.0);
  pose = robotPose(0.8);
  EXPECT_LT(ctrl_->computeVelocityCommands(
      pose, geometry_msgs::msg::Twist(), &goal_checker_).twist.linear.x, 0.0);
}

TEST_F(CandidateEvaluationTest, candidates_preserve_state_and_match_regular_commands)
{
  const auto plan = candidatePlan();
  const auto robot_pose = plan.poses[1];
  const geometry_msgs::msg::Twist speed;
  ctrl_->setPlan(plan);
  ctrl_->setSpeedLimit(0.7, false);
  const double goal_tolerance = ctrl_->getGoalTolerance();
  goal_checker_.tolerance = 0.02;

  const auto normal = ctrl_->evaluateVelocityCommands(plan, 0.7, robot_pose, speed, &goal_checker_);
  const auto avoidance = ctrl_->evaluateVelocityCommands(
    candidatePlan(-0.5), 0.1, robot_pose, speed, &goal_checker_);
  EXPECT_FALSE(normal.collision_detected);
  EXPECT_FALSE(avoidance.collision_detected);
  EXPECT_DOUBLE_EQ(normal.command.twist.linear.x, 0.7);
  EXPECT_DOUBLE_EQ(avoidance.command.twist.linear.x, 0.1);
  EXPECT_GT(normal.command.twist.angular.z, 0.0);
  EXPECT_LT(avoidance.command.twist.angular.z, 0.0);
  EXPECT_EQ(ctrl_->getPlan(), plan);
  EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 0.7);
  EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), goal_tolerance);
  ASSERT_EQ(normal.transformed_plan.poses.size(), 3u);
  EXPECT_EQ(plan.poses.size(), 4u);

  const auto regular = ctrl_->computeVelocityCommands(robot_pose, speed, &goal_checker_);
  EXPECT_EQ(regular, normal.command);
  EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), 0.02);
  EXPECT_EQ(ctrl_->getPlan().poses.size(), 3u);
}

TEST_F(CandidateEvaluationTest, rotation_braking_preserves_candidate_state_and_angular_limit)
{
  update(
  {
    rclcpp::Parameter("test_rpp.use_rotate_to_heading", true),
    rclcpp::Parameter("test_rpp.max_angular_accel", 2.0)});
  goal_checker_.tolerance = 0.05;

  for (double direction : {-1.0, 1.0}) {
    for (double angular_limit : {-1.0, 0.2}) {
      SCOPED_TRACE(direction);
      SCOPED_TRACE(angular_limit);
      auto plan = candidatePlan(0.0);
      plan.poses[2].pose.position.x = 0.01;
      plan.poses[3].pose.position.x = 0.02;
      plan.poses.back().pose.orientation.z = std::sin(direction / 128.0);
      plan.poses.back().pose.orientation.w = std::cos(direction / 128.0);
      const auto robot_pose = plan.poses[1];
      geometry_msgs::msg::Twist speed;
      speed.angular.z = direction * 0.3;
      update({rclcpp::Parameter("test_rpp.max_angular_vel", angular_limit)});
      ctrl_->setPlan(plan);
      ctrl_->setSpeedLimit(0.7, false);
      const double goal_tolerance = ctrl_->getGoalTolerance();

      const auto candidate = ctrl_->evaluateVelocityCommands(
        plan, 0.1, robot_pose, speed, &goal_checker_);
      EXPECT_FALSE(candidate.collision_detected);
      EXPECT_DOUBLE_EQ(candidate.command.twist.linear.x, 0.0);
      EXPECT_NEAR(
        candidate.command.twist.angular.z,
        direction * (angular_limit < 0.0 ? 0.25 : 0.2), 1e-12);
      EXPECT_EQ(ctrl_->getPlan(), plan);
      EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 0.7);
      EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), goal_tolerance);

      const auto regular = ctrl_->computeVelocityCommands(robot_pose, speed, &goal_checker_);
      EXPECT_EQ(regular, candidate.command);
      EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), 0.05);
      EXPECT_EQ(ctrl_->getPlan().poses.size(), 3u);
    }
  }
}

TEST_F(CandidateEvaluationTest, both_entrypoints_hold_costmap_during_evaluation)
{
  const auto plan = candidatePlan();
  const auto robot_pose = plan.poses[1];
  CostmapLockGoalChecker checker(*costmap_->getCostmap()->getMutex());
  ctrl_->setPlan(plan);
  ctrl_->setSpeedLimit(0.7, false);

  const auto candidate = ctrl_->evaluateVelocityCommands(
    plan, 0.7, robot_pose, geometry_msgs::msg::Twist(), &checker);
  const auto regular = ctrl_->computeVelocityCommands(
    robot_pose, geometry_msgs::msg::Twist(), &checker);
  EXPECT_EQ(regular, candidate.command);
  EXPECT_EQ(checker.locked_during_evaluation, (std::vector<bool>{true, true}));

  // PlannerMgmt also holds this recursive lock around a complete planning cycle.
  // Both APIs must remain usable when their caller already owns the map.
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> outer_lock(
    *costmap_->getCostmap()->getMutex());
  EXPECT_NO_THROW(
    ctrl_->evaluateVelocityCommands(
      plan, 0.7, robot_pose, geometry_msgs::msg::Twist(), &goal_checker_));
  EXPECT_NO_THROW(
    ctrl_->computeVelocityCommands(robot_pose, geometry_msgs::msg::Twist(), &goal_checker_));
}

TEST_F(CandidateEvaluationTest, errors_do_not_replace_the_adopted_plan_or_limits)
{
  const auto plan = candidatePlan();
  ctrl_->setPlan(plan);
  ctrl_->setSpeedLimit(0.7, false);
  const auto tolerance = ctrl_->getGoalTolerance();
  goal_checker_.tolerance = 0.02;
  nav_msgs::msg::Path empty_plan;
  EXPECT_THROW(
    ctrl_->evaluateVelocityCommands(
      empty_plan, 0.1, plan.poses[1], geometry_msgs::msg::Twist(), &goal_checker_),
    nav2_core::PlannerException);
  for (double invalid : {-0.1, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    EXPECT_THROW(
      ctrl_->evaluateVelocityCommands(
        plan, invalid, plan.poses[1], geometry_msgs::msg::Twist(), &goal_checker_),
      nav2_core::PlannerException);
  }
  EXPECT_EQ(ctrl_->getPlan(), plan);
  EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 0.7);
  EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), tolerance);
}

TEST_F(CandidateEvaluationTest, absolute_candidate_limits_include_zero_and_override_minimum_speeds)
{
  update(
  {
    rclcpp::Parameter("test_rpp.use_regulated_linear_velocity_scaling", true),
    rclcpp::Parameter("test_rpp.regulated_linear_scaling_min_radius", 100.0),
    rclcpp::Parameter("test_rpp.regulated_linear_scaling_min_speed", 0.25),
    rclcpp::Parameter("test_rpp.min_approach_linear_velocity", 0.2)});
  const auto plan = candidatePlan();
  ctrl_->setPlan(plan);
  ctrl_->setSpeedLimit(0.7, false);
  const double tolerance = ctrl_->getGoalTolerance();
  for (double limit : {0.0, 0.05, 0.1}) {
    const auto result = ctrl_->evaluateVelocityCommands(
      plan, limit, plan.poses[1], geometry_msgs::msg::Twist(), &goal_checker_);
    EXPECT_DOUBLE_EQ(result.command.twist.linear.x, limit);
    EXPECT_DOUBLE_EQ(result.command.twist.angular.z, 2.0 * limit);
    EXPECT_EQ(ctrl_->getPlan(), plan);
    EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 0.7);
    EXPECT_DOUBLE_EQ(ctrl_->getGoalTolerance(), tolerance);
  }
  // The Nav2 setSpeedLimit entry point retains its existing sentinel semantics.
  ctrl_->setSpeedLimit(nav2_costmap_2d::NO_SPEED_LIMIT, false);
  EXPECT_DOUBLE_EQ(ctrl_->getSpeed(), 1.0);
}

TEST_F(CandidateEvaluationTest, collision_results_are_separate_from_other_errors)
{
  ctrl_->deactivate();
  ctrl_->cleanup();
  update({rclcpp::Parameter("test_rpp.use_collision_detection", true)});
  ctrl_->configure(node_, "test_rpp", tf_, costmap_);
  ctrl_->activate();
  auto * map = costmap_->getCostmap();
  unsigned int wall_x, wall_y;
  ASSERT_TRUE(map->worldToMap(0.4, 0.0, wall_x, wall_y));
  for (unsigned int y = 0; y < map->getSizeInCellsY(); ++y) {
    map->setCost(wall_x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  const auto plan = candidatePlan(0.0);
  const auto robot_pose = plan.poses[1];
  const geometry_msgs::msg::Twist speed;
  ctrl_->setPlan(plan);
  const auto evaluation = ctrl_->evaluateVelocityCommands(
    plan, 0.7, robot_pose, speed, &goal_checker_);
  EXPECT_TRUE(evaluation.collision_detected);
  EXPECT_EQ(ctrl_->getPlan(), plan);
  EXPECT_FALSE(evaluation.collision_arc.poses.empty());
  EXPECT_NO_THROW(ctrl_->publishEvaluation(evaluation));
  EXPECT_THROW(
    ctrl_->computeVelocityCommands(robot_pose, speed, &goal_checker_),
    nav2_regulated_pure_pursuit_controller::CollisionDetected);

  try {
    ctrl_->evaluateVelocityCommands(
      nav_msgs::msg::Path(), 0.7, robot_pose, speed, &goal_checker_);
    FAIL() << "An empty plan must fail";
  } catch (const nav2_regulated_pure_pursuit_controller::CollisionDetected &) {
    FAIL() << "An empty plan is not a collision";
  } catch (const nav2_core::PlannerException &) {
    SUCCEED();
  }
}
