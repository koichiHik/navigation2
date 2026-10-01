// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "rcl/time.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"

namespace nav2_costmap_2d
{
class TestDecayLayer : public ObstacleDecayLayer
{
public:
  std::shared_ptr<ObservationBufferWithBase> buffer()
  {
    return observation_buffers_.at(0);
  }
};

class ObstacleDecayFreshness : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("obstacles.observation_sources", "camera"),
      rclcpp::Parameter("obstacles.footprint_clearing_enabled", false),
      rclcpp::Parameter("obstacles.camera.data_type", "PointCloud2"),
      rclcpp::Parameter("obstacles.camera.observation_persistence", 0.0),
      rclcpp::Parameter("obstacles.camera.expected_update_rate", 0.3),
      rclcpp::Parameter("obstacles.camera.max_observation_age", 0.3),
      rclcpp::Parameter("obstacles.camera.max_obstacle_height", 2.0),
      rclcpp::Parameter("obstacles.camera.obstacle_max_range", 10.0)});
    node_ = std::make_shared<nav2_util::LifecycleNode>("decay_freshness_test", "", options);
    node_->declare_parameter("track_unknown_space", false);
    node_->declare_parameter("transform_tolerance", 0.0);
    ASSERT_EQ(RCL_RET_OK, rcl_enable_ros_time_override(node_->get_clock()->get_clock_handle()));
    setTime(100000);
    tf_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    map_ = std::make_unique<LayeredCostmap>("map", true, false);
    map_->resizeMap(10, 10, 1.0, 0.0, 0.0);
    layer_ = std::make_shared<TestDecayLayer>();
    layer_->initialize(map_.get(), "obstacles", tf_.get(), node_, nullptr);
    map_->addPlugin(layer_);
    layer_->activate();
  }

  void TearDown() override
  {
    layer_->deactivate();
    map_.reset();
    layer_.reset();
    tf_.reset();
    node_.reset();
  }

  void setTime(int64_t milliseconds)
  {
    ASSERT_EQ(RCL_RET_OK, rcl_set_ros_time_override(
        node_->get_clock()->get_clock_handle(), milliseconds * 1000000LL));
  }

  sensor_msgs::msg::PointCloud2 cloud(int64_t capture_ms, bool empty = false)
  {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.frame_id = "map";
    msg.header.stamp = rclcpp::Time(capture_ms * 1000000LL, RCL_ROS_TIME);
    msg.height = 1;
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(empty ? 0 : 1);
    if (!empty) {
      sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
      *x = 3.2f;
      *y = 3.2f;
      *z = 0.5f;
    }
    return msg;
  }

  void receive(int64_t capture_ms, bool empty = false)
  {
    layer_->pointCloud2Callback(
      std::make_shared<sensor_msgs::msg::PointCloud2>(cloud(capture_ms, empty)), layer_->buffer());
  }

  void update(double robot_x = 5.0) {map_->updateMap(robot_x, 5.0, 0.0);}

  unsigned char obstacleCost()
  {
    unsigned int mx, my;
    EXPECT_TRUE(map_->getCostmap()->worldToMap(3.2, 3.2, mx, my));
    return map_->getCostmap()->getCost(mx, my);
  }

  void initializeObstacle()
  {
    receive(99900);
    update();
    ASSERT_TRUE(map_->isCurrent());
    ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }

  nav2_util::LifecycleNode::SharedPtr node_;
  std::unique_ptr<tf2_ros::Buffer> tf_;
  std::unique_ptr<LayeredCostmap> map_;
  std::shared_ptr<TestDecayLayer> layer_;
};

TEST_F(ObstacleDecayFreshness, StartupRequiresAnObservation)
{
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_FALSE(map_->isCurrent());
  receive(99950, true);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, Capture216MillisecondsOldStillMarksObstacle)
{
  receive(99784);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100080);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, Capture300MillisecondsOldIsUnavailableWithoutMapUpdate)
{
  initializeObstacle();
  setTime(100199);
  EXPECT_TRUE(map_->isCurrent());
  setTime(100200);
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, FreshArrivalCannotRejuvenateAnUnappliedMap)
{
  initializeObstacle();
  setTime(100250);
  receive(100240, true);
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, FreshEmptyInputClearsAnObstacle)
{
  initializeObstacle();
  setTime(100010);
  receive(100000, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, StaleFutureAndOutOfOrderInputsCannotClear)
{
  initializeObstacle();
  for (const int64_t stamp : {99700, 100001, 99899, 99900}) {
    receive(stamp, true);
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }
  receive(99990, true);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, MalformedInputRetainsMapAndRecoveryRequiresApplyingNewInput)
{
  initializeObstacle();
  auto malformed = cloud(99950, true);
  malformed.fields.clear();
  EXPECT_NO_THROW(layer_->buffer()->bufferCloud(malformed));
  EXPECT_FALSE(map_->isCurrent());
  receive(99990);
  // The previous applied map is still young, but belongs to the pre-error generation.
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update();
  EXPECT_TRUE(map_->isCurrent());
}

TEST_F(ObstacleDecayFreshness, MalformedLayoutAndNonfiniteCoordinatesDoNotClear)
{
  initializeObstacle();
  auto malformed = cloud(99950);
  malformed.data.pop_back();
  EXPECT_NO_THROW(layer_->buffer()->bufferCloud(malformed));
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  auto nonfinite = cloud(99960);
  sensor_msgs::PointCloud2Iterator<float> x(nonfinite, "x");
  *x = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NO_THROW(layer_->buffer()->bufferCloud(nonfinite));
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, TransformFailureRetainsMap)
{
  initializeObstacle();
  auto missing_tf = cloud(99950, true);
  missing_tf.header.frame_id = "unconnected_camera";
  EXPECT_NO_THROW(layer_->buffer()->bufferCloud(missing_tf));
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, RollingWindowPreservesUnavailableObstacleAtWorldPosition)
{
  initializeObstacle();
  setTime(101000);
  update(6.0);
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, ResetCannotPretendToHaveReceivedInput)
{
  initializeObstacle();
  layer_->reset();
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_FALSE(map_->isCurrent());
  receive(99990);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(ObstacleDecayFreshness, ReactivationRequiresNewInput)
{
  initializeObstacle();
  layer_->deactivate();
  layer_->activate();
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99990);
  update();
  EXPECT_TRUE(map_->isCurrent());
}

TEST_F(ObstacleDecayFreshness, LatestObservationRemainsBufferedAfterLongSilence)
{
  initializeObstacle();
  setTime(110000);
  std::vector<Observation> observations;
  layer_->buffer()->getObservations(observations);
  ASSERT_EQ(1u, observations.size());
  EXPECT_EQ(1u, observations.front().cloud_->width);
  EXPECT_FALSE(layer_->buffer()->isCurrent());
}

TEST_F(ObstacleDecayFreshness, ReceiptAgeCanInvalidateIndependentlyOfCaptureAge)
{
  ObservationBufferWithBase buffer(node_, "camera", 0.0, 0.05, 0.0, 2.0,
    10.0, 0.0, 10.0, 0.0, *tf_, "map", "", tf2::durationFromSec(0.0), 0.3);
  buffer.bufferCloud(cloud(99990));
  EXPECT_TRUE(buffer.isCurrent());
  setTime(100050);
  EXPECT_FALSE(buffer.isCurrent());
}
}  // namespace nav2_costmap_2d
