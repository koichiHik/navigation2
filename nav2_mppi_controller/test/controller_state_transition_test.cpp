// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include "gtest/gtest.h"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>

#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>

#include "nav2_mppi_controller/controller.hpp"

#include "utils/utils.hpp"

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

// Tests basic transition from configure->active->process->deactive->cleanup

TEST(ControllerStateTransitionTest, ControllerNotFail)
{
  const bool visualize = true;
  TestCostmapSettings costmap_settings{};

  // Node Options
  rclcpp::NodeOptions options;
  std::vector<rclcpp::Parameter> params;
  setUpControllerParams(visualize, params);
  options.parameter_overrides(params);

  auto node = getDummyNode(options);
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap_ros = getDummyCostmapRos(costmap_settings);
  costmap_ros->setRobotFootprint(getDummySquareFootprint(0.01));

  auto controller = getDummyController(node, tf_buffer, costmap_ros);

  TestPose start_pose = costmap_settings.getCenterPose();
  const double path_step = costmap_settings.resolution;
  TestPathSettings path_settings{start_pose, 8, path_step, path_step};

  // evalControl args
  auto pose = getDummyPointStamped(node, start_pose);
  auto velocity = getDummyTwist();
  auto path = getIncrementalDummyPath(node, path_settings);

  controller->setPlan(path);

  EXPECT_NO_THROW(controller->computeVelocityCommands(pose, velocity, {}));

  controller->setSpeedLimit(0.5, true);
  controller->setSpeedLimit(0.5, false);
  controller->setSpeedLimit(1.0, true);
  controller->deactivate();
  controller->cleanup();
}

class ControllerResetTester : public nav2_mppi_controller::MPPIController
{
public:
  void expireResetPeriod()
  {
    last_time_called_ = clock_->now() - rclcpp::Duration::from_seconds(reset_period_ + 1.0);
  }
};

class ControllerSpeedLimitTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::vector<rclcpp::Parameter> params;
    setUpControllerParams(false, params, "mppic");
    params.emplace_back("mppic.batch_size", 8);
    params.emplace_back("mppic.time_steps", 12);
    params.emplace_back("mppic.critics", std::vector<std::string>{});
    // Equal limits make the resulting command deterministic despite sampled noise.
    // A short horizon also bypasses the unrelated Savitzky-Golay smoothing step.
    params.emplace_back("mppic.vx_min", 0.5);
    params.emplace_back("mppic.vx_max", 0.5);
    params.emplace_back("mppic.wz_max", 0.0);
    rclcpp::NodeOptions options;
    options.parameter_overrides(params);
    node_ = getDummyNode(options);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    costmap_ = getDummyCostmapRos(settings_);
    controller_ = std::make_shared<ControllerResetTester>();
    controller_->configure(node_, "mppic", tf_buffer_, costmap_);
    controller_->activate();
  }

  void TearDown() override
  {
    controller_->deactivate();
    controller_->cleanup();
  }

  double commandVelocity()
  {
    auto start = settings_.getCenterPose();
    auto pose = getDummyPointStamped(node_, start);
    auto path = getIncrementalDummyPath(
      node_, TestPathSettings{start, 8, settings_.resolution, 0.0});
    controller_->setPlan(path);
    return controller_->computeVelocityCommands(pose, getDummyTwist(), nullptr).twist.linear.x;
  }

  TestCostmapSettings settings_;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  std::shared_ptr<ControllerResetTester> controller_;
};

TEST_F(ControllerSpeedLimitTest, ExplicitResetPreservesLimit)
{
  controller_->setSpeedLimit(50.0, true);
  controller_->reset();
  EXPECT_NEAR(commandVelocity(), 0.25, 1e-6);
}

TEST_F(ControllerSpeedLimitTest, IdleResetPreservesLimit)
{
  controller_->setSpeedLimit(50.0, true);
  controller_->expireResetPeriod();
  EXPECT_NEAR(commandVelocity(), 0.25, 1e-6);
}

TEST_F(ControllerSpeedLimitTest, OtherNamespacesPreserveLimitAndOwnUpdatesReset)
{
  controller_->setSpeedLimit(50.0, true);
  node_->declare_parameter("goal_checker.trans_stopped_velocity", 0.01);
  node_->declare_parameter("mppic_other.temperature", 0.3);
  auto result = node_->set_parameters_atomically(
    {rclcpp::Parameter("goal_checker.trans_stopped_velocity", 0.02),
      rclcpp::Parameter("mppic_other.temperature", 0.4),
      rclcpp::Parameter("dummy.verbose", true)});
  ASSERT_TRUE(result.successful);
  EXPECT_NEAR(commandVelocity(), 0.25, 1e-6);

  // An update to this plugin's registered dynamic parameter still resets to base.
  result = node_->set_parameter(rclcpp::Parameter("mppic.temperature", 0.4));
  ASSERT_TRUE(result.successful);
  EXPECT_NEAR(commandVelocity(), 0.5, 1e-6);
}
