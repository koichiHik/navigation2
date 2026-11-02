// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "gtest/gtest.h"
#include "pluginlib/class_loader.hpp"
#include "rcl/time.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"
#include "nav2_costmap_2d/region_aware_decay_layer.hpp"

namespace nav2_costmap_2d
{
namespace
{
using XYZ = std::array<float, 3>;
using Parameters = std::vector<rclcpp::Parameter>;

XYZ obstacle() {return {{3.225f, 3.225f, 0.5f}};}

template<typename LayerType>
class TestLayer : public LayerType
{
public:
  std::shared_ptr<ObservationBufferWithBase> buffer()
  {
    return this->observation_buffers_.at(0);
  }

  double persistenceAt(double x, double y)
  {
    unsigned int mx = 0, my = 0;
    EXPECT_TRUE(this->worldToMap(x, y, mx, my));
    return this->cells_.at(this->getIndex(mx, my)).persistence;
  }

  std::vector<int64_t> deadlinesAt(double x, double y)
  {
    unsigned int mx = 0, my = 0;
    EXPECT_TRUE(this->worldToMap(x, y, mx, my));
    return this->cells_.at(this->getIndex(mx, my)).hold_deadlines_ns;
  }

  bool allHistoriesCleared() const
  {
    return std::all_of(this->cells_.begin(), this->cells_.end(), [](const auto & cell) {
        return !cell.occupied && !cell.latest_hit && cell.last_seen_ns == 0 &&
               cell.persistence == 0.0 && cell.hold_deadlines_ns.empty() &&
               cell.a.values.empty() && cell.a.next == 0 && cell.a.count == 0 &&
               cell.b.values.empty() && cell.b.next == 0 && cell.b.count == 0;
      });
  }

  bool gridIs(unsigned char value)
  {
    const auto * grid = this->getCharMap();
    const size_t size = static_cast<size_t>(this->getSizeInCellsX()) * this->getSizeInCellsY();
    return std::all_of(grid, grid + size, [value](unsigned char cost) {return cost == value;});
  }

  size_t queuedObservations() const {return this->queue_.size();}
};

Parameters region(
  double hold_seconds, const std::string & name = "rear",
  double x_min = -2.1, double x_max = -1.4,
  double y_min = -2.1, double y_max = -1.4)
{
  const std::string prefix = "obstacles.blind_region." + name + ".";
  return {
    rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{name}),
    rclcpp::Parameter(prefix + "x_min", x_min),
    rclcpp::Parameter(prefix + "x_max", x_max),
    rclcpp::Parameter(prefix + "y_min", y_min),
    rclcpp::Parameter(prefix + "y_max", y_max),
    rclcpp::Parameter(prefix + "hold_seconds", hold_seconds)};
}

void append(Parameters & destination, const Parameters & additions)
{
  destination.insert(destination.end(), additions.begin(), additions.end());
}

Parameters adaptivePersistence(
  double tau_s = 1.0, double alpha = 0.12,
  double min_hold_seconds = 0.0, double gamma = 2.0)
{
  return {
    rclcpp::Parameter("obstacles.blind_persistence.enabled", true),
    rclcpp::Parameter("obstacles.blind_persistence.tau_s", tau_s),
    rclcpp::Parameter("obstacles.blind_persistence.alpha", alpha),
    rclcpp::Parameter("obstacles.blind_persistence.min_hold_seconds", min_hold_seconds),
    rclcpp::Parameter("obstacles.blind_persistence.gamma", gamma)};
}

std::vector<geometry_msgs::msg::Point> rectangularFootprint()
{
  std::vector<geometry_msgs::msg::Point> result;
  for (const auto & xy : std::vector<std::array<double, 2>>{
      {{0.21, 0.27}}, {{-0.45, 0.27}}, {{-0.45, -0.27}}, {{0.21, -0.27}}})
  {
    geometry_msgs::msg::Point point;
    point.x = xy[0];
    point.y = xy[1];
    result.push_back(point);
  }
  return result;
}
}  // namespace

class RegionAwareDecayLayerTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override {configure();}
  void TearDown() override {release();}

  static Parameters defaults()
  {
    return {
      rclcpp::Parameter("obstacles.observation_sources", "camera"),
      rclcpp::Parameter("obstacles.footprint_clearing_enabled", false),
      rclcpp::Parameter("obstacles.combination_method", 0),
      rclcpp::Parameter("obstacles.camera.data_type", "PointCloud2"),
      rclcpp::Parameter("obstacles.camera.observation_persistence", 0.0),
      rclcpp::Parameter("obstacles.camera.expected_update_rate", 0.3),
      rclcpp::Parameter("obstacles.camera.max_observation_age", 0.3),
      rclcpp::Parameter("obstacles.camera.min_obstacle_height", 0.0),
      rclcpp::Parameter("obstacles.camera.max_obstacle_height", 2.0),
      rclcpp::Parameter("obstacles.camera.obstacle_max_range", 10.0),
      rclcpp::Parameter("obstacles.camera.obstacle_min_range", 0.0),
      rclcpp::Parameter("obstacles.occupy_min_hits", 1),
      rclcpp::Parameter("obstacles.occupy_window", 1),
      rclcpp::Parameter("obstacles.free_min_misses", 1),
      rclcpp::Parameter("obstacles.free_window", 1)};
  }

  static Parameters mergedParameters(const Parameters & additions)
  {
    std::map<std::string, rclcpp::Parameter> merged;
    for (const auto & parameter : defaults()) {
      merged.emplace(parameter.get_name(), parameter);
    }
    for (const auto & parameter : additions) {
      merged.insert_or_assign(parameter.get_name(), parameter);
    }
    Parameters result;
    for (const auto & entry : merged) {
      result.push_back(entry.second);
    }
    return result;
  }

  void configure(
    const Parameters & additions = {}, bool rolling = true, bool track_unknown = false)
  {
    release();
    rclcpp::NodeOptions options;
    options.parameter_overrides(mergedParameters(additions));
    node_ = std::make_shared<nav2_util::LifecycleNode>(
      "region_aware_decay_layer_test", "", options);
    node_->declare_parameter("track_unknown_space", track_unknown);
    node_->declare_parameter("transform_tolerance", 0.0);
    ASSERT_EQ(RCL_RET_OK, rcl_enable_ros_time_override(node_->get_clock()->get_clock_handle()));
    setTime(100000);
    tf_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    // All transforms are supplied synchronously by the fixture; no listener
    // thread or executor is needed to satisfy these deterministic lookups.
    tf_->setUsingDedicatedThread(true);
    map_ = std::make_unique<LayeredCostmap>("map", rolling, track_unknown);
    map_->resizeMap(200, 200, 0.05, 0.0, 0.0);
    map_->setFootprint(rectangularFootprint());
    layer_ = std::make_shared<TestLayer<RegionAwareDecayLayer>>();
    layer_->initialize(map_.get(), "obstacles", tf_.get(), node_, nullptr);
    map_->addPlugin(layer_);
    layer_->activate();
  }

  void release()
  {
    if (layer_) {
      layer_->deactivate();
    }
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

  sensor_msgs::msg::PointCloud2 cloud(
    int64_t capture_ms, const std::vector<XYZ> & points,
    const std::string & frame = "map")
  {
    sensor_msgs::msg::PointCloud2 message;
    message.header.frame_id = frame;
    message.header.stamp = rclcpp::Time(capture_ms * 1000000LL, RCL_ROS_TIME);
    message.height = 1;
    sensor_msgs::PointCloud2Modifier modifier(message);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(message, "x"), y(message, "y"), z(message, "z");
    for (const auto & point : points) {
      *x = point[0];
      *y = point[1];
      *z = point[2];
      ++x;
      ++y;
      ++z;
    }
    return message;
  }

  void receiveMessage(const sensor_msgs::msg::PointCloud2 & message)
  {
    layer_->pointCloud2Callback(
      std::make_shared<sensor_msgs::msg::PointCloud2>(message), layer_->buffer());
  }

  void receive(int64_t capture_ms, bool empty = false)
  {
    receiveMessage(cloud(capture_ms, empty ? std::vector<XYZ>{} : std::vector<XYZ>{obstacle()}));
  }

  void update(double robot_x = 5.0, double robot_y = 5.0, double robot_yaw = 0.0)
  {
    map_->updateMap(robot_x, robot_y, robot_yaw);
  }

  unsigned char costAt(double x, double y)
  {
    unsigned int mx = 0, my = 0;
    EXPECT_TRUE(map_->getCostmap()->worldToMap(x, y, mx, my));
    return map_->getCostmap()->getCost(mx, my);
  }

  unsigned char obstacleCost() {return costAt(obstacle()[0], obstacle()[1]);}

  double persistence() {return layer_->persistenceAt(obstacle()[0], obstacle()[1]);}

  std::vector<int64_t> deadlines()
  {
    return layer_->deadlinesAt(obstacle()[0], obstacle()[1]);
  }

  void expectNoPersistence()
  {
    EXPECT_DOUBLE_EQ(0.0, persistence());
    const auto values = deadlines();
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](int64_t value) {return value == 0;}));
  }

  void initializeObstacle()
  {
    receive(99900);
    update();
    ASSERT_TRUE(map_->isCurrent());
    ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }

  void clearWorldRectangle(double low, double high, bool invert)
  {
    int low_x, low_y, high_x, high_y;
    layer_->worldToMapEnforceBounds(low, low, low_x, low_y);
    layer_->worldToMapEnforceBounds(high, high, high_x, high_y);
    layer_->clearArea(low_x, low_y, high_x, high_y, invert);
  }

  nav2_util::LifecycleNode::SharedPtr node_;
  std::unique_ptr<tf2_ros::Buffer> tf_;
  std::unique_ptr<LayeredCostmap> map_;
  std::shared_ptr<TestLayer<RegionAwareDecayLayer>> layer_;
};

TEST_F(RegionAwareDecayLayerTest, StartupRequiresAnAppliedValidObservation)
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

TEST_F(RegionAwareDecayLayerTest, TwoQueuedHitsCountAsTwoObservationsInOneCostmapUpdate)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  receive(99850);
  receive(99950);
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, OneHitThenMissDoesNotMeetTwoOfThreeOccupancy)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  receive(99850);
  receive(99900, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99950);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, OccupancyWindowExpiresOlderHits)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  receive(99800);
  receive(99830, true);
  receive(99860, true);
  receive(99900, true);
  receive(99950);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99975);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, TwoQueuedEmptyCloudsClearAndOneMissRetains)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  receive(99920, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99940, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());

  receive(99950);
  update();
  receive(99960, true);
  receive(99970, true);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, CasePOldHitsCannotReoccupyAFreeCell)
{
  configure({rclcpp::Parameter("obstacles.occupy_window", 3)});
  const std::vector<unsigned char> expected{
    LETHAL_OBSTACLE, FREE_SPACE, FREE_SPACE, LETHAL_OBSTACLE};
  for (size_t i = 0; i < expected.size(); ++i) {
    receive(99800 + 40 * i, i == 1 || i == 2);
    update();
    EXPECT_EQ(expected[i], obstacleCost()) << "observation " << i + 1;
  }
}

TEST_F(RegionAwareDecayLayerTest, CaseQDestinationHistoryStartsEmptyAndDoesNotResetWhileStable)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 5)});
  const std::vector<unsigned char> expected{
    FREE_SPACE, FREE_SPACE, FREE_SPACE, LETHAL_OBSTACLE, LETHAL_OBSTACLE, FREE_SPACE};
  for (size_t i = 0; i < expected.size(); ++i) {
    receive(99800 + 30 * i, i < 2 || i >= 4);
    update();
    EXPECT_EQ(expected[i], obstacleCost()) << "observation " << i + 1;
  }
}

TEST_F(RegionAwareDecayLayerTest, RepeatedUpdatesDoNotInventHitsOrMisses)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  receive(99850);
  update();
  for (int i = 0; i < 4; ++i) {
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
  }
  receive(99900);
  update();
  receive(99950, true);
  update();
  for (int i = 0; i < 4; ++i) {
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, RawDuplicateInvalidatesFreshnessWithoutCountingTwice)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  receive(99900);
  update();
  receive(99900);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99950);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, MissingAndStaleInputsRetainAnObstacle)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  setTime(100199);
  EXPECT_TRUE(map_->isCurrent());
  setTime(100200);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100190, true);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, QueuedCloudThatAgesOutDoesNotCountAsAMiss)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  receive(100000, true);
  setTime(100301);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100290, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100300, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, QueueOverflowIsUnhealthyAndRequiresNewObservation)
{
  configure({rclcpp::Parameter("obstacles.observation_queue_size", 2),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  receive(99920, true);
  receive(99940, true);
  receive(99960, true);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99980, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99990, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, InvalidCloudAndTransformFailureDoNotBecomeMisses)
{
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
      rclcpp::Parameter("obstacles.free_window", 3)});
    initializeObstacle();
    auto invalid = cloud(99920, {obstacle()});
    if (kind == 0) {
      invalid.fields.clear();
    } else if (kind == 1) {
      invalid.data.pop_back();
    } else if (kind == 2) {
      sensor_msgs::PointCloud2Iterator<float> x(invalid, "x");
      *x = std::numeric_limits<float>::quiet_NaN();
    } else {
      invalid.header.frame_id = "unconnected_camera";
    }
    EXPECT_NO_THROW(receiveMessage(invalid));
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    receive(99950, true);
    EXPECT_FALSE(map_->isCurrent());
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    receive(99975, true);
    update();
    EXPECT_EQ(FREE_SPACE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, StaleFutureAndOutOfOrderCloudsCannotClear)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  for (const int64_t stamp : {99700, 100001, 99899, 99900}) {
    receive(stamp, true);
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }
  receive(99950, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, ZeroHoldStillRequiresTheMissThreshold)
{
  auto parameters = region(0.0);
  append(parameters, {rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  configure(parameters);
  initializeObstacle();
  receive(99930, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99960, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, FiniteHoldNeedsANewMissCapturedAfterItsDeadline)
{
  auto parameters = region(0.3);
  append(parameters, {rclcpp::Parameter("obstacles.camera.expected_update_rate", 1.0),
    rclcpp::Parameter("obstacles.camera.max_observation_age", 1.0)});
  configure(parameters);
  initializeObstacle();
  receive(99950, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100199);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100200);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100250);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100240, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, StaleInputRemainsUnavailableUntilANewValidObservation)
{
  configure(region(0.3));
  initializeObstacle();
  receive(99950, true);
  update();
  setTime(100250);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100240, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, QueuedMissUsesItsCaptureTimeBeforeALaterHitRefreshesHold)
{
  auto parameters = region(0.15);
  append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.camera.expected_update_rate", 0.5),
    rclcpp::Parameter("obstacles.camera.max_observation_age", 0.5)});
  configure(parameters);
  setTime(100020);
  receive(100000);
  receive(100010);
  update();
  ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());

  setTime(100200);
  // At capture 100.050 the last hit (100.010) is only 40 ms old. Draining
  // at 100.200 must not expire it before processing the later hit at 100.180:
  // that would clear the cell and require two new hits to reoccupy it.
  receive(100050, true);
  receive(100180);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());

  receive(100190, true);
  update();
  setTime(100329);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100330);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100330, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, ANewHitRefreshesHoldAndWinsOverOldMisses)
{
  auto parameters = region(0.2);
  append(parameters, {rclcpp::Parameter("obstacles.free_window", 3),
    rclcpp::Parameter("obstacles.camera.expected_update_rate", 1.0),
    rclcpp::Parameter("obstacles.camera.max_observation_age", 1.0)});
  configure(parameters);
  initializeObstacle();
  receive(99950, true);
  update();
  receive(99990);
  update();
  setTime(100250);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100240, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, LeavingBlindRegionByTranslationOrYawNeedsANewMissToFree)
{
  for (bool rotate : {false, true}) {
    configure(region(-1.0));
    initializeObstacle();
    receive(99950, true);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    if (rotate) {
      update(5.0, 5.0, std::acos(-1.0) / 2.0);
    } else {
      update(3.0, 5.0);
    }
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    receive(99975, true);
    if (rotate) {
      update(5.0, 5.0, std::acos(-1.0) / 2.0);
    } else {
      update(3.0, 5.0);
    }
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, OverlappingRegionsUseLongestHoldAndInfinityWins)
{
  for (double longer : {0.5, -1.0}) {
    auto parameters = region(0.0, "short");
    append(parameters, region(longer, "long"));
    append(parameters, {
      rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"short", "long"}),
      rclcpp::Parameter("obstacles.camera.expected_update_rate", 10.0),
      rclcpp::Parameter("obstacles.camera.max_observation_age", 10.0)});
    configure(parameters);
    initializeObstacle();
    receive(99950, true);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    setTime(100500);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    receive(100490, true);
    update();
    EXPECT_EQ(longer < 0.0 ? LETHAL_OBSTACLE : FREE_SPACE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, RollingWindowKeepsWorldPositionAndDropsOutOfWindowHistory)
{
  configure(region(-1.0, "all", -100.0, 100.0, -100.0, 100.0));
  initializeObstacle();
  receive(99950, true);
  update(6.0, 5.0);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  EXPECT_EQ(FREE_SPACE, costAt(obstacle()[0] + 1.0, obstacle()[1]));
  update(6.0, 6.0, 1.5);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(101000);
  update(7.0, 6.0, -1.5);
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update(25.0, 25.0);
  unsigned int mx, my;
  EXPECT_FALSE(map_->getCostmap()->worldToMap(obstacle()[0], obstacle()[1], mx, my));
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(100990, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, RollingWindowMovesPartialHitAndMissHistoriesWithTheirCell)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  receive(99850);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99900);
  update(6.0, 5.0);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99930, true);
  update(6.0, 5.0);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99960, true);
  update(6.0, 6.0);
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, BlindRegionUsesCellCenterRatherThanPartialCellOverlap)
{
  // The obstacle cell spans x=[3.20, 3.25]; relative to robot x=5 this
  // rectangle intersects that cell but excludes its center at x=-1.775.
  configure(region(-1.0, "thin", -1.79, -1.78, -1.9, -1.6));
  initializeObstacle();
  receive(99950, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, BaseLinkCloudIsTransformedExactlyOnceAndRemainsWorldFixed)
{
  configure(region(-1.0, "all", -100.0, 100.0, -100.0, 100.0));
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.header.stamp = rclcpp::Time(99900000000LL, RCL_ROS_TIME);
  transform.child_frame_id = "base_link";
  transform.transform.translation.x = 5.0;
  transform.transform.translation.y = 5.0;
  transform.transform.rotation.z = std::sqrt(0.5);
  transform.transform.rotation.w = std::sqrt(0.5);
  ASSERT_TRUE(tf_->setTransform(transform, "region_aware_decay_layer_test", true));

  // A 90-degree map<-base_link rotation plus (5,5) translation maps (1,0)
  // to (5,6). The shared observation buffer already performs this transform.
  receiveMessage(cloud(99900, {{{1.0f, 0.0f, 0.5f}}}, "base_link"));
  update();
  ASSERT_TRUE(map_->isCurrent());
  ASSERT_EQ(LETHAL_OBSTACLE, costAt(5.0, 6.0));
  auto * grid = map_->getCostmap()->getCharMap();
  EXPECT_EQ(1, std::count(grid, grid + 200 * 200, LETHAL_OBSTACLE));

  receiveMessage(cloud(99950, {}, "base_link"));
  update(6.0, 6.0, 0.7);
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(5.0, 6.0));
  EXPECT_EQ(FREE_SPACE, costAt(6.0, 7.0));
  grid = map_->getCostmap()->getCharMap();
  EXPECT_EQ(1, std::count(grid, grid + 200 * 200, LETHAL_OBSTACLE));
}

TEST_F(RegionAwareDecayLayerTest, RectangularFootprintClearsRearCornersWithoutHistoryRevival)
{
  auto parameters = region(-1.0, "all", -100.0, 100.0, -100.0, 100.0);
  append(parameters, {rclcpp::Parameter("obstacles.footprint_clearing_enabled", true)});
  configure(parameters);
  const std::vector<XYZ> inside{
    {{4.575f, 5.225f, 0.5f}}, {{4.575f, 4.775f, 0.5f}}, {{5.175f, 5.225f, 0.5f}}};
  auto points = inside;
  points.push_back({{4.475f, 5.225f, 0.5f}});
  receiveMessage(cloud(99900, points));
  update(6.0, 5.0);
  for (const auto & point : inside) {
    ASSERT_EQ(LETHAL_OBSTACLE, costAt(point[0], point[1]));
  }
  receive(99950, true);
  update();
  for (const auto & point : inside) {
    EXPECT_EQ(FREE_SPACE, costAt(point[0], point[1]));
  }
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(4.475, 5.225));
  update(6.0, 5.0);
  for (const auto & point : inside) {
    EXPECT_EQ(FREE_SPACE, costAt(point[0], point[1]));
  }
}

TEST_F(RegionAwareDecayLayerTest, ClearAreaAndItsInverseClearHistoryAsWellAsGrid)
{
  for (bool invert : {false, true}) {
    configure(region(-1.0, "all", -100.0, 100.0, -100.0, 100.0));
    receiveMessage(cloud(99900, {obstacle(), {{6.225f, 6.225f, 0.5f}}}));
    update();
    ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
    ASSERT_EQ(LETHAL_OBSTACLE, costAt(6.225, 6.225));
    clearWorldRectangle(2.0, 4.0, invert);
    update();
    EXPECT_EQ(invert ? LETHAL_OBSTACLE : FREE_SPACE, obstacleCost());
    EXPECT_EQ(invert ? FREE_SPACE : LETHAL_OBSTACLE, costAt(6.225, 6.225));
    receive(99950, true);
    update();
    EXPECT_EQ(invert ? LETHAL_OBSTACLE : FREE_SPACE, obstacleCost());
    EXPECT_EQ(invert ? FREE_SPACE : LETHAL_OBSTACLE, costAt(6.225, 6.225));
  }
}

TEST_F(RegionAwareDecayLayerTest, PartialClearPreservesEveryQueuedOutsideHitAndItsPointCount)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.obstacle_pointcount", 2)});
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  // Removing the two middle points must preserve both outside points in each
  // cloud, rather than dropping a whole sample or combining counts across it.
  const std::vector<XYZ> points{outside, obstacle(), obstacle(), outside};
  receiveMessage(cloud(99900, points));
  receiveMessage(cloud(99950, points));
  clearWorldRectangle(2.0, 4.0, false);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
}

TEST_F(RegionAwareDecayLayerTest, PartialClearRetainsOutsideCellsWithoutReprocessingOldInput)
{
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  receiveMessage(cloud(99900, {obstacle(), outside}));
  update();
  ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  ASSERT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  clearWorldRectangle(2.0, 4.0, false);
  EXPECT_FALSE(map_->isCurrent());
  for (int i = 0; i < 3; ++i) {
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  }
  update(6.0, 5.0, 0.5);
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
}

TEST_F(RegionAwareDecayLayerTest, PartialClearAndInverseRespectAllFourStrictCellBoundaries)
{
  const std::vector<XYZ> boundary{
    {{2.025f, 3.225f, 0.5f}}, {{4.025f, 3.225f, 0.5f}},
    {{3.225f, 2.025f, 0.5f}}, {{3.225f, 4.025f, 0.5f}}};
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  auto points = boundary;
  points.push_back(obstacle());
  points.push_back(outside);
  for (bool invert : {false, true}) {
    configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
      rclcpp::Parameter("obstacles.occupy_window", 3)});
    receiveMessage(cloud(99900, points));
    receiveMessage(cloud(99950, points));
    // Cell coordinates exactly on start/end are outside the strict interior.
    layer_->clearArea(40, 40, 80, 80, invert);
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(invert ? LETHAL_OBSTACLE : FREE_SPACE, obstacleCost());
    for (const auto & point : boundary) {
      EXPECT_EQ(invert ? FREE_SPACE : LETHAL_OBSTACLE, costAt(point[0], point[1]));
    }
    EXPECT_EQ(invert ? FREE_SPACE : LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  }
}

TEST_F(RegionAwareDecayLayerTest, PartialClearPreservesQueuedEmptyCloudsForOutsideMissCounting)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  initializeObstacle();
  receive(99930, true);
  receive(99960, true);
  // Clearing a distant area must not discard valid free evidence at (3.225,3.225).
  clearWorldRectangle(6.0, 8.0, false);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, FullyFilteredCloudsStillCountAsOutsideMisses)
{
  configure({rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  receiveMessage(cloud(99900, {obstacle(), outside}));
  update();
  ASSERT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  // These nonempty observations become empty when their only target point is
  // masked. They must still contribute two misses at the untouched outside cell.
  receive(99930);
  receive(99960);
  clearWorldRectangle(2.0, 4.0, false);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(FREE_SPACE, costAt(outside[0], outside[1]));
}

TEST_F(RegionAwareDecayLayerTest, PartialClearDoesNotRenewCaptureOrReceiptFreshness)
{
  for (bool capture_limit : {false, true}) {
    configure({
      rclcpp::Parameter("obstacles.camera.max_observation_age", capture_limit ? 0.3 : 1.0),
      rclcpp::Parameter("obstacles.camera.expected_update_rate", capture_limit ? 1.0 : 0.3)});
    receive(99900);
    update();
    ASSERT_TRUE(map_->isCurrent());
    setTime(100150);
    clearWorldRectangle(6.0, 8.0, false);
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    const int64_t expiry = capture_limit ? 100200 : 100300;
    setTime(expiry - 1);
    EXPECT_TRUE(map_->isCurrent());
    setTime(expiry);
    EXPECT_FALSE(map_->isCurrent());
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, PartialClearPreservesSourceOrderingAndNewInputCanReoccupy)
{
  initializeObstacle();
  clearWorldRectangle(2.0, 4.0, false);
  update();
  ASSERT_TRUE(map_->isCurrent());
  ASSERT_EQ(FREE_SPACE, obstacleCost());
  // An explicit clear must not reset source stamp/generation validation.
  receive(99900);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99950);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, RepeatedPartialClearsDoNotReviveQueuedPointsAfterRolling)
{
  auto parameters = region(-1.0, "all", -100.0, 100.0, -100.0, 100.0);
  append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  configure(parameters);
  const XYZ second{{4.225f, 4.225f, 0.5f}};
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  const std::vector<XYZ> points{obstacle(), second, outside};
  receiveMessage(cloud(99900, points));
  receiveMessage(cloud(99950, points));
  clearWorldRectangle(2.0, 3.8, false);
  clearWorldRectangle(4.0, 4.8, false);
  update(6.0, 6.0, 0.7);
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(FREE_SPACE, costAt(second[0], second[1]));
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  receive(99975, true);
  update(7.0, 7.0, -0.7);
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(FREE_SPACE, costAt(second[0], second[1]));
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  clearWorldRectangle(6.0, 7.0, false);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_EQ(FREE_SPACE, costAt(second[0], second[1]));
  EXPECT_EQ(FREE_SPACE, costAt(outside[0], outside[1]));
}

TEST_F(RegionAwareDecayLayerTest, PartialClearBetweenBoundsAndCostsPreservesPendingFreshnessCommit)
{
  const XYZ outside{{6.225f, 6.225f, 0.5f}};
  for (bool previous_sample : {false, true}) {
    SCOPED_TRACE(previous_sample);
    configure();
    if (previous_sample) {
      initializeObstacle();
    }
    setTime(100250);
    ASSERT_FALSE(map_->isCurrent());
    receiveMessage(cloud(100240, {obstacle(), outside}));

    // Prepare fresh B without committing it. Either there is no applied sample,
    // or the previously applied A has already expired; B is the only way to
    // make this update current after a service clears its target cells.
    double min_x = std::numeric_limits<double>::max();
    double min_y = min_x;
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = max_x;
    layer_->updateBounds(5.0, 5.0, 0.0, &min_x, &min_y, &max_x, &max_y);
    EXPECT_FALSE(map_->isCurrent());
    clearWorldRectangle(2.0, 4.0, false);

    // LayeredCostmap resets the master update region between these two phases.
    map_->getCostmap()->resetMap(0, 0, 200, 200);
    layer_->updateCosts(*map_->getCostmap(), 0, 0, 200, 200);
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    EXPECT_EQ(LETHAL_OBSTACLE, costAt(outside[0], outside[1]));
  }
}

TEST_F(RegionAwareDecayLayerTest, SizeChangesAndExplicitResetDiscardHistoryAndQueuedClouds)
{
  for (bool resize : {false, true}) {
    configure(region(-1.0, "all", -100.0, 100.0, -100.0, 100.0));
    initializeObstacle();
    receive(99950);
    if (resize) {
      map_->resizeMap(100, 100, 0.1, 0.0, 0.0);
    } else {
      layer_->reset();
    }
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    receive(99975, true);
    update();
    EXPECT_TRUE(map_->isCurrent());
    EXPECT_EQ(FREE_SPACE, obstacleCost());
  }
}

TEST_F(RegionAwareDecayLayerTest, LifecycleCallsImmediatelyClearGridHistoriesAndQueuedObservations)
{
  const XYZ partial{{4.225f, 3.225f, 0.5f}};
  for (bool adaptive : {false, true}) {
    for (bool track_unknown : {false, true}) {
      for (bool activate : {false, true}) {
        SCOPED_TRACE(adaptive);
        SCOPED_TRACE(track_unknown);
        SCOPED_TRACE(activate);
        auto parameters = region(20.0);
        if (adaptive) {
          append(parameters, adaptivePersistence(1.0, 0.5));
        }
        append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 2),
          rclcpp::Parameter("obstacles.occupy_window", 3),
          rclcpp::Parameter("obstacles.free_min_misses", 2),
          rclcpp::Parameter("obstacles.free_window", 3)});
        configure(parameters, true, track_unknown);
        receive(99900);
        receive(99920);
        receiveMessage(cloud(99940, {partial}));
        update();
        ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
        ASSERT_EQ(FREE_SPACE, costAt(partial[0], partial[1]));
        ASSERT_TRUE(map_->isCurrent());
        ASSERT_FALSE(layer_->allHistoriesCleared());
        ASSERT_FALSE(layer_->gridIs(track_unknown ? NO_INFORMATION : FREE_SPACE));
        receiveMessage(cloud(99960, {obstacle(), partial}));
        receive(99970, true);
        ASSERT_EQ(2u, layer_->queuedObservations());

        // Test each call independently: a reset in the other lifecycle call
        // must not hide retained grid cells or histories here.
        if (activate) {
          layer_->activate();
        } else {
          layer_->deactivate();
        }
        EXPECT_FALSE(layer_->isCurrent());
        EXPECT_FALSE(map_->isCurrent());
        EXPECT_TRUE(layer_->allHistoriesCleared());
        EXPECT_TRUE(layer_->gridIs(track_unknown ? NO_INFORMATION : FREE_SPACE));
        EXPECT_EQ(0u, layer_->queuedObservations());
        update();
        EXPECT_FALSE(map_->isCurrent());
        EXPECT_TRUE(layer_->allHistoriesCleared());
        EXPECT_TRUE(layer_->gridIs(track_unknown ? NO_INFORMATION : FREE_SPACE));
      }
    }
  }
}

TEST_F(RegionAwareDecayLayerTest, ReactivationRequiresNewHitsToRebuildFixedHoldOccupancy)
{
  auto parameters = region(20.0);
  append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  configure(parameters);
  receive(99900);
  receive(99920);
  receive(99930, true);
  update();
  ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99940);
  receive(99945, true);
  layer_->deactivate();
  layer_->activate();
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99950, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99960);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receive(99975);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  // Outside the blind region, only the two new misses may satisfy b.
  receive(99980, true);
  update(3.0, 5.0);
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(99990, true);
  update(3.0, 5.0);
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, CallbacksWhileDeactivatedCannotRepopulateResetHistory)
{
  initializeObstacle();
  layer_->deactivate();
  receive(99950);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_TRUE(layer_->allHistoriesCleared());
  EXPECT_EQ(0u, layer_->queuedObservations());
  layer_->activate();
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, BackwardClockRequiresNewInputAndDoesNotReviveOldHistory)
{
  configure(region(-1.0, "all", -100.0, 100.0, -100.0, 100.0));
  initializeObstacle();
  receive(99950);
  setTime(90000);
  update();
  EXPECT_FALSE(map_->isCurrent());
  receive(89990, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, PointCountThresholdIsPerCloudNotAcrossQueuedClouds)
{
  configure({rclcpp::Parameter("obstacles.obstacle_pointcount", 2)});
  receive(99900);
  receive(99950);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  receiveMessage(cloud(99975, {obstacle(), obstacle()}));
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, HeightRangeAndMapBoundsFiltersMatchObstacleLayerSemantics)
{
  configure({rclcpp::Parameter("obstacles.camera.obstacle_max_range", 6.0),
    rclcpp::Parameter("obstacles.camera.obstacle_min_range", 1.0)});
  receiveMessage(cloud(99900, {
      obstacle(), {{4.225f, 3.225f, 3.0f}}, {{4.225f, 4.225f, -0.1f}},
      {{6.225f, 0.225f, 0.5f}}, {{0.225f, 0.225f, 0.5f}}, {{-1.225f, 1.225f, 0.5f}}}));
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  const auto * grid = map_->getCostmap()->getCharMap();
  EXPECT_EQ(1, std::count(grid, grid + 200 * 200, LETHAL_OBSTACLE));
}

TEST_F(RegionAwareDecayLayerTest, InvalidParametersAreRejectedAtInitialization)
{
  std::vector<Parameters> invalid{
    {rclcpp::Parameter("obstacles.occupy_min_hits", 0)},
    {rclcpp::Parameter("obstacles.occupy_min_hits", 2)},
    {rclcpp::Parameter("obstacles.occupy_window", 1025)},
    {rclcpp::Parameter("obstacles.free_min_misses", 0)},
    {rclcpp::Parameter("obstacles.free_min_misses", 2)},
    {rclcpp::Parameter("obstacles.free_window", 1025)},
    {rclcpp::Parameter("obstacles.observation_queue_size", 0)},
    {rclcpp::Parameter("obstacles.obstacle_pointcount", 0)},
    {rclcpp::Parameter("obstacles.camera.observation_persistence", 1.0)},
    {rclcpp::Parameter("obstacles.camera.expected_update_rate", 0.0)},
    {rclcpp::Parameter("obstacles.camera.max_observation_age", 0.0)},
    {rclcpp::Parameter("obstacles.camera.clearing", true)},
    {rclcpp::Parameter("obstacles.camera.marking", false)},
    {rclcpp::Parameter("obstacles.camera.data_type", "LaserScan")},
    {rclcpp::Parameter("obstacles.observation_sources", "camera second")},
    region(-0.5), region(1.0, "reversed", 1.0, -1.0),
    region(std::numeric_limits<double>::infinity()),
    region(1.0, "nonfinite", std::numeric_limits<double>::quiet_NaN())};
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_ANY_THROW(configure(invalid[i]));
  }
}

TEST_F(RegionAwareDecayLayerTest, NewHistoryConfigurationCannotBeChangedHalfwayThroughAnObservation)
{
  const auto result = node_->set_parameters_atomically(
    {rclcpp::Parameter("obstacles.occupy_window", 2)});
  EXPECT_FALSE(result.successful);
  initializeObstacle();
}

TEST_F(RegionAwareDecayLayerTest, PersistenceCountsOneHitPerCloudBeforeOccupancy)
{
  auto parameters = adaptivePersistence();
  append(parameters, region(20.0));
  append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 3),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  configure(parameters);
  receiveMessage(cloud(99790, {obstacle(), obstacle(), obstacle()}));
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_DOUBLE_EQ(0.12, persistence());
  const auto before_occupancy = deadlines();
  EXPECT_TRUE(std::all_of(
      before_occupancy.begin(), before_occupancy.end(), [](int64_t value) {return value == 0;}));

  receive(99890);
  update();
  const double second = 0.12 * std::exp(-0.1) + 0.12;
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_NEAR(second, persistence(), 1e-12);
  receive(99990);
  update();
  const double third = second * std::exp(-0.1) + 0.12;
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  EXPECT_NEAR(third, persistence(), 1e-12);
  ASSERT_EQ(1u, deadlines().size());
  EXPECT_NEAR(99990000000LL + 20.0 * third * third * 1e9, deadlines()[0], 2.0);
}

TEST_F(RegionAwareDecayLayerTest, PersistenceDependsOnHitIntervalsAndSaturatesAtOne)
{
  std::vector<double> scores;
  for (const int64_t interval_ms : {100, 500}) {
    configure(adaptivePersistence());
    for (int hit = 0; hit < 3; ++hit) {
      const int64_t stamp_ms = 99990 + hit * interval_ms;
      setTime(stamp_ms + 10);
      receive(stamp_ms);
      update();
    }
    const double retained = std::exp(-interval_ms / 1000.0);
    EXPECT_NEAR(0.12 * (1.0 + retained + retained * retained), persistence(), 1e-12);
    scores.push_back(persistence());
  }
  EXPECT_GT(scores[0], scores[1]);

  configure(adaptivePersistence(1.0, 0.8));
  receive(99900);
  receive(99950);
  update();
  EXPECT_DOUBLE_EQ(1.0, persistence());
}

TEST_F(RegionAwareDecayLayerTest, PersistenceIgnoresRetainedGridAndRejectedHits)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(20.0));
  configure(parameters);
  initializeObstacle();
  const auto original_deadlines = deadlines();
  for (const int64_t invalid_stamp : {99900, 99899, 100001}) {
    receive(invalid_stamp);
    update();
    EXPECT_FALSE(map_->isCurrent());
    EXPECT_DOUBLE_EQ(0.5, persistence());
    EXPECT_EQ(original_deadlines, deadlines());
  }
  receive(99950, true);
  update();
  setTime(101000);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  EXPECT_DOUBLE_EQ(0.5, persistence());
  EXPECT_EQ(original_deadlines, deadlines());
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveDeadlinesArePreparedOutsideRegionsAndOverlapUsesMaximum)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(0.4, "short"));
  append(parameters, region(2.0, "long"));
  append(parameters, {
    rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"short", "long"})});
  configure(parameters);
  receive(99900);
  // The hit is outside both blind regions. Prepare both deadlines anyway.
  update(3.0, 5.0);
  ASSERT_EQ(2u, deadlines().size());
  EXPECT_NEAR(100000000000LL, deadlines()[0], 2.0);
  EXPECT_NEAR(100400000000LL, deadlines()[1], 2.0);
  const auto prepared = deadlines();

  setTime(100100);
  receive(100090, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  EXPECT_EQ(prepared, deadlines());
  // Entering the regions must not start another 0.5-second hold.
  setTime(100420);
  receive(100410, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  expectNoPersistence();
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveHoldUsesOnlyCurrentlyOccupiedRegions)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(2.0, "short", -0.1, 0.6));
  append(parameters, region(8.0, "long"));
  append(parameters, {
    rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"short", "long"})});
  for (bool outside_all : {false, true}) {
    SCOPED_TRACE(outside_all);
    configure(parameters);
    initializeObstacle();
    ASSERT_EQ(2u, deadlines().size());
    EXPECT_GT(deadlines()[1], deadlines()[0]);
    const int64_t miss_ms = outside_all ? 99950 : 100450;
    setTime(std::max(int64_t{100000}, miss_ms + 10));
    receive(miss_ms, true);
    // At x=3 only "short" applies; at x=4 neither region applies. Neither
    // situation may use the still-future deadline prepared for "long".
    update(outside_all ? 4.0 : 3.0, 5.0);
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    expectNoPersistence();
  }
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveHoldInterpolatesFromTheConfiguredMinimum)
{
  for (const double minimum : {0.0, 0.4, 2.0}) {
    SCOPED_TRACE(minimum);
    auto parameters = adaptivePersistence(1.0, 0.5, minimum);
    append(parameters, region(2.0));
    configure(parameters);
    initializeObstacle();
    ASSERT_EQ(1u, deadlines().size());
    const double expected_hold = minimum + (2.0 - minimum) * 0.25;
    EXPECT_NEAR(99900000000LL + expected_hold * 1e9, deadlines()[0], 2.0);
  }
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveHoldPreservesZeroAndInfiniteRegionSemantics)
{
  for (const double hold_seconds : {0.0, -1.0}) {
    SCOPED_TRACE(hold_seconds);
    auto parameters = adaptivePersistence(1.0, 0.5);
    append(parameters, region(hold_seconds, "special"));
    append(parameters, region(2.0, "finite", -0.1, 0.6));
    append(parameters, {
      rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"special", "finite"}),
      rclcpp::Parameter("obstacles.free_min_misses", 2),
      rclcpp::Parameter("obstacles.free_window", 3)});
    configure(parameters);
    initializeObstacle();
    receive(99930, true);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    setTime(103000);
    receive(102990, true);
    update();
    EXPECT_EQ(hold_seconds < 0.0 ? LETHAL_OBSTACLE : FREE_SPACE, obstacleCost());
  }

  // An overlapping infinite region still overrides an expired finite hold.
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(0.4, "finite"));
  append(parameters, region(-1.0, "infinite"));
  append(parameters, {
    rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"finite", "infinite"})});
  configure(parameters);
  initializeObstacle();
  setTime(101000);
  receive(100990, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, RehitCanLowerPersistenceWithoutShorteningDeadlineAndCanExtendIt)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(20.0));
  configure(parameters);
  receive(99990);
  receive(99991);
  receive(99992);
  update();
  ASSERT_DOUBLE_EQ(1.0, persistence());
  ASSERT_EQ(1u, deadlines().size());
  const int64_t original_deadline = deadlines()[0];

  setTime(101002);
  receive(100992);
  update();
  const double lower_score = std::exp(-1.0) + 0.5;
  EXPECT_NEAR(lower_score, persistence(), 1e-12);
  EXPECT_EQ(original_deadline, deadlines()[0]);

  // Even with a lower score, a sufficiently later hit can produce a later
  // absolute deadline. Compare deadlines, rather than comparing scores.
  setTime(119002);
  receive(118992);
  update();
  const double latest_score = lower_score * std::exp(-18.0) + 0.5;
  EXPECT_NEAR(latest_score, persistence(), 1e-12);
  EXPECT_LT(persistence(), lower_score);
  EXPECT_GT(deadlines()[0], original_deadline);
  EXPECT_NEAR(118992000000LL + 20.0 * latest_score * latest_score * 1e9, deadlines()[0], 2.0);
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveExpiryStillNeedsAFreshMissAndTheMissThreshold)
{
  for (const int misses_before_expiry : {0, 2}) {
    SCOPED_TRACE(misses_before_expiry);
    auto parameters = adaptivePersistence(1.0, 0.5);
    append(parameters, region(0.8));
    append(parameters, {rclcpp::Parameter("obstacles.free_min_misses", 2),
      rclcpp::Parameter("obstacles.free_window", 3),
      rclcpp::Parameter("obstacles.camera.expected_update_rate", 1.0),
      rclcpp::Parameter("obstacles.camera.max_observation_age", 1.0)});
    configure(parameters);
    initializeObstacle();
    const auto original_deadlines = deadlines();
    for (int miss = 0; miss < misses_before_expiry; ++miss) {
      receive(99930 + miss * 30, true);
      update();
      EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
      EXPECT_DOUBLE_EQ(0.5, persistence());
      EXPECT_EQ(original_deadlines, deadlines());
    }
    // Deadline is 100.100. Timer updates and even a delayed, pre-deadline
    // capture must not clear the cell using the current timer time.
    setTime(100200);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    receive(100090, true);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    EXPECT_EQ(original_deadlines, deadlines());

    receive(100190, true);
    update();
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    expectNoPersistence();
    // A fresh occupancy must start with alpha; no old score or deadline survives Free.
    receive(100195);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    EXPECT_DOUBLE_EQ(0.5, persistence());
    ASSERT_EQ(1u, deadlines().size());
    EXPECT_NEAR(100395000000LL, deadlines()[0], 2.0);
  }
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveExpiryDoesNotReplaceTheMissThreshold)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(0.4));
  append(parameters, {rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3)});
  configure(parameters);
  initializeObstacle();
  setTime(100100);
  receive(100090, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100095, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, ReactivationStartsAdaptiveScoresAndDeadlinesFromNewHits)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(20.0));
  append(parameters, {rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3)});
  configure(parameters);
  receive(99900);
  receive(99910);
  receive(99920);
  receive(99930, true);
  update();
  ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  ASSERT_DOUBLE_EQ(1.0, persistence());
  ASSERT_EQ(1u, deadlines().size());
  const int64_t previous_deadline = deadlines()[0];
  receive(99940);
  layer_->deactivate();
  EXPECT_TRUE(layer_->allHistoriesCleared());
  EXPECT_TRUE(layer_->gridIs(FREE_SPACE));
  layer_->activate();
  EXPECT_TRUE(layer_->allHistoriesCleared());
  EXPECT_TRUE(layer_->gridIs(FREE_SPACE));
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  expectNoPersistence();
  receive(99950);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  EXPECT_DOUBLE_EQ(0.5, persistence());
  EXPECT_TRUE(deadlines().empty());
  receive(99960);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  const double score = 0.5 * std::exp(-0.01) + 0.5;
  EXPECT_NEAR(score, persistence(), 1e-12);
  ASSERT_EQ(1u, deadlines().size());
  EXPECT_NEAR(99960000000LL + 20.0 * score * score * 1e9, deadlines()[0], 2.0);
  EXPECT_LT(deadlines()[0], previous_deadline);
}

TEST_F(RegionAwareDecayLayerTest, AdaptiveStateIsDiscardedByClearResetResizeClockAndEnableChanges)
{
  for (int action = 0; action < 5; ++action) {
    SCOPED_TRACE(action);
    auto parameters = adaptivePersistence(1.0, 0.5);
    append(parameters, region(20.0));
    configure(parameters);
    initializeObstacle();
    ASSERT_GT(persistence(), 0.0);
    ASSERT_FALSE(deadlines().empty());
    receive(99950);
    int64_t next_stamp = 99975;
    switch (action) {
      case 0:
        clearWorldRectangle(2.0, 4.0, false);
        break;
      case 1:
        layer_->reset();
        break;
      case 2:
        map_->resizeMap(100, 100, 0.1, 0.0, 0.0);
        break;
      case 3:
        setTime(90000);
        next_stamp = 89990;
        break;
      default:
        ASSERT_TRUE(node_->set_parameters_atomically(
            {rclcpp::Parameter("obstacles.enabled", false)}).successful);
        ASSERT_TRUE(node_->set_parameters_atomically(
            {rclcpp::Parameter("obstacles.enabled", true)}).successful);
        break;
    }
    update();
    EXPECT_EQ(FREE_SPACE, obstacleCost());
    expectNoPersistence();
    receive(next_stamp);
    update();
    EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
    EXPECT_DOUBLE_EQ(0.5, persistence());
  }
}

TEST_F(RegionAwareDecayLayerTest, RollingMovesAdaptiveStateWithWorldCellAndDropsItOutsideWindow)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(20.0, "all", -100.0, 100.0, -100.0, 100.0));
  configure(parameters);
  initializeObstacle();
  const auto original_deadlines = deadlines();
  update(6.0, 6.0, 1.5);
  EXPECT_DOUBLE_EQ(0.5, persistence());
  EXPECT_EQ(original_deadlines, deadlines());
  EXPECT_DOUBLE_EQ(0.0, layer_->persistenceAt(obstacle()[0] + 1.0, obstacle()[1] + 1.0));
  receive(99950);
  update(6.0, 6.0);
  EXPECT_NEAR(0.5 * std::exp(-0.05) + 0.5, persistence(), 1e-12);
  update(25.0, 25.0);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
  expectNoPersistence();
  receive(99975);
  update();
  EXPECT_DOUBLE_EQ(0.5, persistence());
}

TEST_F(RegionAwareDecayLayerTest, FootprintClearingDiscardsAdaptiveEvidence)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(20.0, "all", -100.0, 100.0, -100.0, 100.0));
  append(parameters, {rclcpp::Parameter("obstacles.footprint_clearing_enabled", true)});
  configure(parameters);
  const XYZ corner{{4.575f, 5.225f, 0.5f}};
  receiveMessage(cloud(99900, {corner}));
  update(6.0, 5.0);
  ASSERT_EQ(LETHAL_OBSTACLE, costAt(corner[0], corner[1]));
  ASSERT_DOUBLE_EQ(0.5, layer_->persistenceAt(corner[0], corner[1]));
  update();
  EXPECT_EQ(FREE_SPACE, costAt(corner[0], corner[1]));
  EXPECT_DOUBLE_EQ(0.0, layer_->persistenceAt(corner[0], corner[1]));
  const auto cleared_deadlines = layer_->deadlinesAt(corner[0], corner[1]);
  EXPECT_TRUE(std::all_of(
      cleared_deadlines.begin(), cleared_deadlines.end(), [](int64_t value) {return value == 0;}));
  update(6.0, 5.0);
  receiveMessage(cloud(99950, {corner}));
  update(6.0, 5.0);
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(corner[0], corner[1]));
  EXPECT_DOUBLE_EQ(0.5, layer_->persistenceAt(corner[0], corner[1]));
}

TEST_F(RegionAwareDecayLayerTest, DisabledPersistenceKeepsTheFixedRegionHold)
{
  auto parameters = adaptivePersistence(1.0, 0.12);
  append(parameters, region(1.0));
  append(parameters, {rclcpp::Parameter("obstacles.blind_persistence.enabled", false)});
  configure(parameters);
  initializeObstacle();
  receive(99950, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(101000);
  receive(100990, true);
  update();
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, LargeFiniteAdaptiveHoldsSaturateTheDeadlineWithoutOverflow)
{
  auto parameters = adaptivePersistence(1.0, 0.5);
  append(parameters, region(std::numeric_limits<double>::max()));
  configure(parameters);
  initializeObstacle();
  ASSERT_EQ(1u, deadlines().size());
  EXPECT_EQ(std::numeric_limits<int64_t>::max(), deadlines()[0]);
  receive(99950, true);
  update();
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, InvalidPersistenceParametersAreRejectedAtInitialization)
{
  Parameters invalid{
    rclcpp::Parameter("obstacles.blind_persistence.tau_s", 0.0),
    rclcpp::Parameter("obstacles.blind_persistence.tau_s", -1.0),
    rclcpp::Parameter("obstacles.blind_persistence.alpha", 0.0),
    rclcpp::Parameter("obstacles.blind_persistence.alpha", -0.1),
    rclcpp::Parameter("obstacles.blind_persistence.alpha", 1.01),
    rclcpp::Parameter("obstacles.blind_persistence.min_hold_seconds", -0.1),
    rclcpp::Parameter("obstacles.blind_persistence.gamma", 1.0),
    rclcpp::Parameter("obstacles.blind_persistence.gamma", 0.0)};
  for (const std::string name : {"tau_s", "alpha", "min_hold_seconds", "gamma"}) {
    for (double value : {std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()})
    {
      invalid.emplace_back("obstacles.blind_persistence." + name, value);
    }
  }
  for (const auto & parameter : invalid) {
    SCOPED_TRACE(parameter.get_name());
    auto parameters = adaptivePersistence();
    parameters.push_back(parameter);
    EXPECT_ANY_THROW(configure(parameters));
  }
  auto parameters = adaptivePersistence(1.0, 0.12, 0.6);
  append(parameters, region(0.5));
  EXPECT_ANY_THROW(configure(parameters));
  append(parameters, {rclcpp::Parameter("obstacles.blind_persistence.enabled", false)});
  EXPECT_NO_THROW(configure(parameters));

  parameters = adaptivePersistence(1.0, 0.12, 0.4);
  append(parameters, region(1.0, "long"));
  append(parameters, region(0.2, "short"));
  append(parameters, {
    rclcpp::Parameter("obstacles.blind_region_names", std::vector<std::string>{"long", "short"})});
  EXPECT_ANY_THROW(configure(parameters));
}

TEST_F(RegionAwareDecayLayerTest, PersistenceParametersRequireReconfiguration)
{
  configure(adaptivePersistence());
  auto changes = adaptivePersistence(2.0, 0.2, 0.1, 3.0);
  changes.front() = rclcpp::Parameter("obstacles.blind_persistence.enabled", false);
  for (const auto & parameter : changes) {
    const auto result = node_->set_parameters_atomically({parameter});
    EXPECT_FALSE(result.successful) << parameter.get_name();
  }
}

TEST_F(RegionAwareDecayLayerTest, BothOldAndNewClassesAreLoadableThroughPluginlib)
{
  pluginlib::ClassLoader<Layer> loader("nav2_costmap_2d", "nav2_costmap_2d::Layer");
  auto new_instance = loader.createSharedInstance("nav2_costmap_2d::RegionAwareDecayLayer");
  auto old_instance = loader.createSharedInstance("nav2_costmap_2d::ObstacleDecayLayer");
  EXPECT_NE(nullptr, dynamic_cast<RegionAwareDecayLayer *>(new_instance.get()));
  EXPECT_NE(nullptr, dynamic_cast<ObstacleDecayLayer *>(old_instance.get()));
}

TEST_F(RegionAwareDecayLayerTest, OneOfOneModeDoesNotReviveFootprintClearedCellsWithoutNewInput)
{
  configure({rclcpp::Parameter("obstacles.footprint_clearing_enabled", true)});
  const XYZ rear_corner{{4.575f, 5.225f, 0.5f}};
  receiveMessage(cloud(99900, {obstacle(), rear_corner}));
  update();
  ASSERT_TRUE(map_->isCurrent());
  ASSERT_EQ(FREE_SPACE, costAt(rear_corner[0], rear_corner[1]));
  ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());

  // With 1/1 thresholds and no blind regions, the layer must still use saved
  // cell state. Rebuilding the old snapshot would resurrect this cleared point
  // as soon as the footprint moves away from it.
  update(6.0, 5.0);
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, costAt(rear_corner[0], rear_corner[1]));
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update(6.0, 5.0, 0.4);
  EXPECT_EQ(FREE_SPACE, costAt(rear_corner[0], rear_corner[1]));

  receiveMessage(cloud(99950, {rear_corner}));
  update(6.0, 5.0);
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, costAt(rear_corner[0], rear_corner[1]));
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, OneOfOneModeKeepsFreshnessTiedToTheAppliedObservation)
{
  initializeObstacle();
  setTime(100199);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  setTime(100200);
  EXPECT_FALSE(map_->isCurrent());
  receive(100190, true);
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());

  auto malformed = cloud(100195, {});
  malformed.fields.clear();
  receiveMessage(malformed);
  EXPECT_FALSE(map_->isCurrent());
  receive(100199);
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100199);
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, obstacleCost());
  receive(100200, true);
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, obstacleCost());
}

TEST_F(RegionAwareDecayLayerTest, BenchmarkReportsSyntheticTenHertzReplayAtFiveHertzUpdates)
{
  configure({rclcpp::Parameter("obstacles.occupy_min_hits", 2),
    rclcpp::Parameter("obstacles.occupy_window", 3),
    rclcpp::Parameter("obstacles.free_min_misses", 2),
    rclcpp::Parameter("obstacles.free_window", 3),
    rclcpp::Parameter("obstacles.camera.obstacle_max_range", 20.0)});
  std::vector<XYZ> dense;
  dense.reserve(200 * 200);
  for (int y = 0; y < 200; ++y) {
    for (int x = 0; x < 200; ++x) {
      dense.push_back({{0.025f + 0.05f * x, 0.025f + 0.05f * y, 0.5f}});
    }
  }
  std::vector<double> wall_us;
  const auto cpu_start = std::clock();
  for (int cycle = 0; cycle < 25; ++cycle) {
    const int64_t now = 100200 + cycle * 200;
    setTime(now);
    receiveMessage(cloud(now - 100, dense));
    receiveMessage(cloud(now - 1, dense));
    const auto start = std::chrono::steady_clock::now();
    update();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    wall_us.push_back(std::chrono::duration<double, std::micro>(elapsed).count());
    ASSERT_TRUE(map_->isCurrent());
    ASSERT_EQ(LETHAL_OBSTACLE, obstacleCost());
  }
  const double cpu_ms = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
  std::sort(wall_us.begin(), wall_us.end());
  const double median = wall_us[wall_us.size() / 2];
  const double p95 = wall_us[static_cast<size_t>(std::ceil(wall_us.size() * 0.95)) - 1];
  const double maximum = wall_us.back();
  RecordProperty("synthetic_update_median_us", std::to_string(median));
  RecordProperty("synthetic_update_p95_us", std::to_string(p95));
  RecordProperty("synthetic_update_max_us", std::to_string(maximum));
  RecordProperty("synthetic_replay_cpu_ms", std::to_string(cpu_ms));
  std::cout << "200x200 synthetic replay, 40,000 points/cloud, two clouds/update: median="
            << median << " us, p95=" << p95 << " us, max=" << maximum
            << " us, replay CPU=" << cpu_ms << " ms" << std::endl;
  // Real hardware timing and stopping/recovery acceptance still require recorded and live runs.
}
}  // namespace nav2_costmap_2d
