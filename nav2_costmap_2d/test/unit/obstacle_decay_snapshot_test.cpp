// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "camera_localization_msgs/msg/obstacle_grid_snapshot.hpp"
#include "camera_localization_msgs/msg/obstacle_measurement_cell.hpp"
#include "camera_localization_msgs/msg/obstacle_status.hpp"
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"
#include "rcl/time.h"

namespace nav2_costmap_2d
{
namespace
{

using Snapshot = camera_localization_msgs::msg::ObstacleGridSnapshot;
using Cell = camera_localization_msgs::msg::ObstacleMeasurementCell;
using Status = camera_localization_msgs::msg::ObstacleStatus;

builtin_interfaces::msg::Time RosTime(const std::int64_t nanoseconds)
{
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  result.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return result;
}

class ObstacleDecaySnapshotTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  virtual std::int64_t hitCountThreshold() const {return 2;}
  virtual bool rollingWindow() const {return false;}

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
      {rclcpp::Parameter("obstacles.snapshot_input_enabled", true),
        rclcpp::Parameter("obstacles.hit_count_threshold", hitCountThreshold()),
        rclcpp::Parameter("obstacles.unreported_cells_are_free", true),
        rclcpp::Parameter("obstacles.max_status_age", 0.5),
        rclcpp::Parameter("obstacles.max_capture_age", 0.5),
        rclcpp::Parameter("obstacles.footprint_clearing_enabled", true),
        rclcpp::Parameter("obstacles.observation_sources", "legacy_ignored"),
        rclcpp::Parameter("obstacles.combination_method", 0)});
    node_ = std::make_shared<nav2_util::LifecycleNode>("decay_snapshot_test", "", options);
    node_->declare_parameter("track_unknown_space", true);
    node_->declare_parameter("transform_tolerance", 0.0);
    ASSERT_EQ(RCL_RET_OK, rcl_enable_ros_time_override(node_->get_clock()->get_clock_handle()));
    setTimeMilliseconds(100'000);

    tf_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    geometry_msgs::msg::TransformStamped base_to_map;
    base_to_map.header.frame_id = "map";
    base_to_map.child_frame_id = "base_link";
    base_to_map.transform.rotation.w = 1.0;
    ASSERT_TRUE(tf_->setTransform(base_to_map, "snapshot_test", true));

    map_ = std::make_unique<LayeredCostmap>("map", rollingWindow(), true);
    map_->resizeMap(100U, 100U, 0.1, 0.0, 0.0);
    map_->setFootprint(
      {geometry_msgs::msg::Point{}, geometry_msgs::msg::Point{},
        geometry_msgs::msg::Point{}, geometry_msgs::msg::Point{}});
    auto footprint = map_->getFootprint();
    footprint[0].x = -0.45;
    footprint[0].y = -0.30;
    footprint[1].x = 0.30;
    footprint[1].y = -0.30;
    footprint[2].x = 0.30;
    footprint[2].y = 0.30;
    footprint[3].x = -0.45;
    footprint[3].y = 0.30;
    map_->setFootprint(footprint);

    layer_ = std::make_shared<ObstacleDecayLayer>();
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

  void setTimeMilliseconds(const std::int64_t milliseconds)
  {
    ASSERT_EQ(
      RCL_RET_OK, rcl_set_ros_time_override(
        node_->get_clock()->get_clock_handle(),
        milliseconds * 1'000'000LL));
  }

  Snapshot snapshot(
    const std::uint64_t sequence, const bool with_obstacle,
    const std::int32_t cell_x = 32, const std::int32_t cell_y = 32,
    const std::string & frame = "base_link") const
  {
    const std::int64_t capture_ns = (99'900 + static_cast<std::int64_t>(sequence)) * 1'000'000LL;
    const std::int64_t receive_ns = capture_ns + 10'000'000LL;
    Snapshot result;
    result.schema_version = Snapshot::SCHEMA_VERSION;
    result.header.frame_id = frame;
    result.header.stamp = RosTime(capture_ns);
    result.receive_stamp = RosTime(receive_ns);
    result.receive_timestamp_ns = receive_ns;
    result.grid_resolution = 0.1F;
    result.grid_origin_x = 0.0;
    result.grid_origin_y = 0.0;
    result.grid_width = 100U;
    result.grid_height = 100U;
    result.measurement_complete = true;
    result.candidate_empty = true;
    result.cluster_empty = !with_obstacle;
    result.free_space_semantics = Snapshot::FREE_SPACE_HITS_ONLY;

    Status & status = result.status;
    status.schema_version = Status::SCHEMA_VERSION;
    status.status_stamp = result.receive_stamp;
    status.capture_stamp = result.header.stamp;
    status.status_timestamp_ns = receive_ns;
    status.capture_timestamp_ns = capture_ns;
    status.run_generation = 5U;
    status.frame_sequence = sequence;
    status.processing_state = Status::PROCESSING_VALID;
    status.process_alive = true;
    status.image_fresh = true;
    status.physical_stereo_complete = true;
    status.obstacle_complete = true;
    status.calibration_match = true;
    status.sgm_profile_match = true;
    status.authority_valid = true;
    status.required_camera_mask = 1U;
    status.present_camera_mask = 1U;
    status.required_pair_count = 1U;
    status.processed_pair_count = 1U;
    status.point_count = with_obstacle ? 1U : 0U;
    status.occupied_cell_count = with_obstacle ? 1U : 0U;
    status.obstacle_frame_id = frame;

    if (with_obstacle) {
      Cell cell;
      cell.ix = cell_x;
      cell.iy = cell_y;
      cell.z_min = 0.1F;
      cell.z_max = 0.8F;
      cell.stage = Cell::STAGE_CLUSTER_ACCEPTED;
      cell.point_count = 5U;
      cell.quality = 0.8F;
      cell.contributing_source_mask = 1U;
      result.cluster_accepted_cells.push_back(cell);
    }
    return result;
  }

  void receive(const Snapshot & value)
  {
    layer_->obstacleGridSnapshotCallback(std::make_shared<Snapshot>(value));
  }

  void update(
    const double robot_x = 5.0, const double robot_y = 5.0,
    const double robot_yaw = 0.0)
  {
    map_->updateMap(robot_x, robot_y, robot_yaw);
  }

  unsigned char cost(const double x, const double y)
  {
    unsigned int mx = 0U;
    unsigned int my = 0U;
    EXPECT_TRUE(map_->getCostmap()->worldToMap(x, y, mx, my));
    return map_->getCostmap()->getCost(mx, my);
  }

  void setDynamicTransform(
    const std::string & child_frame, const std::int64_t timestamp_ns,
    const double x, const double y)
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "map";
    transform.child_frame_id = child_frame;
    transform.header.stamp = RosTime(timestamp_ns);
    transform.transform.translation.x = x;
    transform.transform.translation.y = y;
    transform.transform.rotation.w = 1.0;
    ASSERT_TRUE(tf_->setTransform(transform, "snapshot_test", false));
  }

  nav2_util::LifecycleNode::SharedPtr node_;
  std::unique_ptr<tf2_ros::Buffer> tf_;
  std::unique_ptr<LayeredCostmap> map_;
  std::shared_ptr<ObstacleDecayLayer> layer_;
};

class OneHitSnapshotTest : public ObstacleDecaySnapshotTest
{
protected:
  std::int64_t hitCountThreshold() const override {return 1;}
};

class ThreeHitSnapshotTest : public ObstacleDecaySnapshotTest
{
protected:
  std::int64_t hitCountThreshold() const override {return 3;}
};

class RollingSnapshotTest : public ObstacleDecaySnapshotTest
{
protected:
  bool rollingWindow() const override {return true;}
};

TEST_F(ObstacleDecaySnapshotTest, DefaultRequiresTwoDistinctFrames) {
  receive(snapshot(1U, true));
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);

  receive(snapshot(2U, true));
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(OneHitSnapshotTest, ThresholdOneMarksFirstFrame) {
  receive(snapshot(1U, true));
  update();
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ThreeHitSnapshotTest, ThresholdThreeWaitsForThirdFrame) {
  receive(snapshot(1U, true));
  update();
  receive(snapshot(2U, true));
  update();
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);
  receive(snapshot(3U, true));
  update();
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, ExactSequenceRetryDoesNotAddHit) {
  const Snapshot first = snapshot(1U, true);
  receive(first);
  update();
  receive(first);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);
  receive(snapshot(2U, true));
  update();
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, NewGenerationDoesNotInheritPendingHit) {
  receive(snapshot(1U, true));
  update();
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);

  Snapshot restarted_first = snapshot(1U, true);
  restarted_first.status.run_generation = 6U;
  receive(restarted_first);
  update();
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);

  Snapshot restarted_second = snapshot(2U, true);
  restarted_second.status.run_generation = 6U;
  receive(restarted_second);
  update();
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, ValidMissRetainsOnceThenClears) {
  receive(snapshot(1U, true));
  update();
  receive(snapshot(2U, true));
  update();
  ASSERT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
  receive(snapshot(3U, false));
  update();
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
  receive(snapshot(4U, false));
  update();
  EXPECT_EQ(cost(3.25, 3.25), FREE_SPACE);
}

TEST_F(ObstacleDecaySnapshotTest, StopWithSameSequenceRevokesCurrentAndHoldsEvidence) {
  receive(snapshot(1U, true));
  update();
  Snapshot second = snapshot(2U, true);
  receive(second);
  update();
  ASSERT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
  ASSERT_TRUE(map_->isCurrent());

  second.status.processing_state = Status::PROCESSING_STOPPED;
  second.status.authority_valid = false;
  receive(second);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, InvalidHighSequenceDoesNotPoisonValidRecovery) {
  receive(snapshot(1U, true));
  update();
  receive(snapshot(2U, true));
  update();
  ASSERT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);

  Snapshot invalid = snapshot(100U, false);
  invalid.status.authority_valid = false;
  receive(invalid);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);

  receive(snapshot(3U, false));
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, StaleInputHoldsEvidenceAndBecomesNotCurrent) {
  receive(snapshot(1U, true));
  update();
  receive(snapshot(2U, true));
  update();
  ASSERT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
  setTimeMilliseconds(101'000);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, MissingTransformHoldsEvidenceAndBecomesNotCurrent) {
  receive(snapshot(1U, true));
  update();
  receive(snapshot(2U, true));
  update();
  ASSERT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);

  receive(snapshot(3U, false, 32, 32, "missing_frame"));
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(cost(3.25, 3.25), LETHAL_OBSTACLE);
}

TEST_F(ObstacleDecaySnapshotTest, CurrentFootprintMasksButDoesNotEraseEvidence) {
  receive(snapshot(1U, true, 50, 50));
  update();
  receive(snapshot(2U, true, 50, 50));
  update();
  EXPECT_EQ(cost(5.05, 5.05), FREE_SPACE);

  update(6.0, 5.0);
  EXPECT_EQ(cost(5.05, 5.05), LETHAL_OBSTACLE);
}

TEST_F(RollingSnapshotTest, RobotMotionMapsDifferentLocalCellsToSameWorldEvidence) {
  Snapshot first = snapshot(1U, true, 32, 0, "moving_base");
  first.grid_height = 10U;
  first.cluster_accepted_cells.front().iy = 0;
  setDynamicTransform("moving_base", first.status.capture_timestamp_ns, 5.0, 5.0);
  receive(first);
  update(5.0, 5.0);
  EXPECT_EQ(cost(8.25, 5.05), FREE_SPACE);

  Snapshot second = snapshot(2U, true, 22, 0, "moving_base");
  second.grid_height = 10U;
  second.cluster_accepted_cells.front().iy = 0;
  setDynamicTransform("moving_base", second.status.capture_timestamp_ns, 6.0, 5.0);
  receive(second);
  update(6.0, 5.0);
  EXPECT_DOUBLE_EQ(map_->getCostmap()->getOriginX(), 1.0);
  EXPECT_EQ(cost(8.25, 5.05), LETHAL_OBSTACLE);
}

TEST_F(RollingSnapshotTest, ForwardOffsetSharesOriginAndCoversRequiredEnvelope) {
  constexpr double kResolution = 0.1;
  constexpr double kForwardOffset = 5.0;
  constexpr double kRobotX = 42.0;
  constexpr double kRobotY = -17.0;
  constexpr double kPi = 3.14159265358979323846;
  map_->resizeMap(200U, 200U, kResolution, 0.0, 0.0);
  map_->setRollingWindowForwardOffset(kForwardOffset);

  Costmap2D * const master = map_->getCostmap();
  for (int degrees = 0; degrees < 360; ++degrees) {
    SCOPED_TRACE(degrees);
    const double yaw = static_cast<double>(degrees) * kPi / 180.0;
    const double forward_x = std::cos(yaw);
    const double forward_y = std::sin(yaw);
    const double left_x = -forward_y;
    const double left_y = forward_x;
    update(kRobotX, kRobotY, yaw);

    EXPECT_DOUBLE_EQ(master->getOriginX(), layer_->getOriginX());
    EXPECT_DOUBLE_EQ(master->getOriginY(), layer_->getOriginY());

    const double requested_center_x = kRobotX + kForwardOffset * forward_x;
    const double requested_center_y = kRobotY + kForwardOffset * forward_y;
    const double actual_center_x =
      master->getOriginX() + master->getSizeInMetersX() / 2.0;
    const double actual_center_y =
      master->getOriginY() + master->getSizeInMetersY() / 2.0;
    EXPECT_LE(std::abs(actual_center_x - requested_center_x), kResolution + 1e-12);
    EXPECT_LE(std::abs(actual_center_y - requested_center_y), kResolution + 1e-12);

    const auto expect_inside = [&](const double forward_distance, const double left_distance) {
        const double x = kRobotX + forward_distance * forward_x + left_distance * left_x;
        const double y = kRobotY + forward_distance * forward_y + left_distance * left_y;
        unsigned int mx = 0U;
        unsigned int my = 0U;
        EXPECT_TRUE(master->worldToMap(x, y, mx, my));
        EXPECT_TRUE(layer_->worldToMap(x, y, mx, my));
      };
    expect_inside(14.8, 0.0);
    expect_inside(-4.8, 0.0);
    expect_inside(0.0, 8.4);
    expect_inside(0.0, -8.4);
  }
}

TEST_F(RollingSnapshotTest, ForwardOffsetRejectsNan) {
  map_->setRollingWindowForwardOffset(5.0);
  EXPECT_THROW(
    map_->setRollingWindowForwardOffset(std::numeric_limits<double>::quiet_NaN()),
    std::invalid_argument);

  update(10.0, 10.0, 0.0);
  const double actual_center_x =
    map_->getCostmap()->getOriginX() + map_->getCostmap()->getSizeInMetersX() / 2.0;
  EXPECT_NEAR(actual_center_x, 15.0, map_->getCostmap()->getResolution());
}

TEST_F(ObstacleDecaySnapshotTest, SnapshotParametersAreStartupOnly) {
  const auto result = layer_->dynamicParametersCallback(
    {rclcpp::Parameter("obstacles.hit_count_threshold", std::int64_t{3})});
  EXPECT_FALSE(result.successful);
  EXPECT_EQ(result.reason, "snapshot obstacle parameters are startup-only");
}

}  // namespace
}  // namespace nav2_costmap_2d
