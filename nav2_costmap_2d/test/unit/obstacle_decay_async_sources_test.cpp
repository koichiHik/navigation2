// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "gtest/gtest.h"
#include "rcl/time.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"
#include "nav2_costmap_2d/region_aware_decay_layer.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"

namespace nav2_costmap_2d
{
namespace
{
using XYZ = std::array<float, 3>;
enum class DecayKind {Snapshot, RegionAware};

template<typename LayerType>
class IndependentDecayLayer : public LayerType
{
public:
  std::shared_ptr<ObservationBufferWithBase> buffer()
  {
    return this->observation_buffers_.at(0);
  }
};
}  // namespace

// Both sources remain required for navigation. Separate MAX layers allow one
// source to update while the other is unavailable, without claiming the map is
// current or interpreting a missing observation as an empty obstacle cloud.
class ObstacleDecayAsyncSources : public ::testing::TestWithParam<DecayKind>
{
protected:
  enum Source : std::size_t {Ess, Gemini};

  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override {configure();}
  void TearDown() override {release();}

  void configure(
    bool inflation = false, const std::vector<rclcpp::Parameter> & additions = {})
  {
    release();
    std::vector<rclcpp::Parameter> parameters;
    for (const auto & name : {"ess", "gemini"}) {
      const std::string prefix = std::string(name) + ".";
      const std::vector<rclcpp::Parameter> source_parameters{
        rclcpp::Parameter(prefix + "observation_sources", "camera"),
        rclcpp::Parameter(prefix + "combination_method", 1),
        rclcpp::Parameter(prefix + "footprint_clearing_enabled", false),
        rclcpp::Parameter(prefix + "camera.topic", std::string("/") + name + "/points"),
        rclcpp::Parameter(prefix + "camera.data_type", "PointCloud2"),
        rclcpp::Parameter(prefix + "camera.observation_persistence", 0.0),
        rclcpp::Parameter(prefix + "camera.expected_update_rate", 0.3),
        rclcpp::Parameter(prefix + "camera.max_observation_age", 0.3),
        rclcpp::Parameter(prefix + "camera.max_obstacle_height", 2.0),
        rclcpp::Parameter(prefix + "camera.obstacle_max_range", 10.0)};
      parameters.insert(parameters.end(), source_parameters.begin(), source_parameters.end());
    }
    parameters.emplace_back("inflation.inflation_radius", 1.5);
    parameters.emplace_back("inflation.cost_scaling_factor", 1.0);
    std::map<std::string, rclcpp::Parameter> merged;
    for (const auto & parameter : parameters) {
      merged.insert_or_assign(parameter.get_name(), parameter);
    }
    for (const auto & parameter : additions) {
      merged.insert_or_assign(parameter.get_name(), parameter);
    }
    parameters.clear();
    for (const auto & entry : merged) {
      parameters.push_back(entry.second);
    }
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameters);
    node_ = std::make_shared<nav2_util::LifecycleNode>("decay_async_sources_test", "", options);
    node_->declare_parameter("track_unknown_space", false);
    node_->declare_parameter("transform_tolerance", 0.0);
    ASSERT_EQ(RCL_RET_OK, rcl_enable_ros_time_override(node_->get_clock()->get_clock_handle()));
    setTime(100000);
    tf_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
    tf_->setUsingDedicatedThread(true);
    map_ = std::make_unique<LayeredCostmap>("map", true, false);
    map_->resizeMap(10, 10, 1.0, 0.0, 0.0);
    const std::array<std::string, 2> names{{"ess", "gemini"}};
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      if (GetParam() == DecayKind::Snapshot) {
        addSource<ObstacleDecayLayer>(i, names[i]);
      } else {
        addSource<RegionAwareDecayLayer>(i, names[i]);
      }
    }
    if (inflation) {
      inflation_ = std::make_shared<InflationLayer>();
      inflation_->initialize(map_.get(), "inflation", tf_.get(), node_, nullptr);
      map_->addPlugin(inflation_);  // Inflation must see the completed two-source union.
      std::vector<geometry_msgs::msg::Point> footprint;
      for (const auto & xy : std::array<std::array<double, 2>, 4>{{
          {{-0.2, -0.2}}, {{0.2, -0.2}}, {{0.2, 0.2}}, {{-0.2, 0.2}}}})
      {
        geometry_msgs::msg::Point point;
        point.x = xy[0];
        point.y = xy[1];
        footprint.push_back(point);
      }
      map_->setFootprint(footprint);
    }
  }

  template<typename LayerType>
  void addSource(std::size_t index, const std::string & name)
  {
    auto layer = std::make_shared<IndependentDecayLayer<LayerType>>();
    layer->initialize(map_.get(), name, tf_.get(), node_, nullptr);
    map_->addPlugin(layer);
    layer->activate();
    layers_[index] = layer;
    receive_[index] = [layer](const sensor_msgs::msg::PointCloud2 & msg) {
        layer->pointCloud2Callback(
          std::make_shared<sensor_msgs::msg::PointCloud2>(msg), layer->buffer());
      };
  }

  void release()
  {
    for (auto & layer : layers_) {
      if (layer) {
        layer->deactivate();
      }
    }
    map_.reset();
    inflation_.reset();
    for (auto & callback : receive_) {
      callback = {};
    }
    for (auto & layer : layers_) {
      layer.reset();
    }
    tf_.reset();
    node_.reset();
  }

  void setTime(int64_t milliseconds)
  {
    ASSERT_EQ(RCL_RET_OK, rcl_set_ros_time_override(
        node_->get_clock()->get_clock_handle(), milliseconds * 1000000LL));
  }

  sensor_msgs::msg::PointCloud2 cloud(
    int64_t capture_ms, std::initializer_list<XYZ> points,
    const std::string & frame = "map")
  {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.frame_id = frame;
    msg.header.stamp = rclcpp::Time(capture_ms * 1000000LL, RCL_ROS_TIME);
    msg.height = 1;
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());
    if (points.size() == 0) {
      return msg;
    }
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    for (const auto & point : points) {
      *x = point[0];
      *y = point[1];
      *z = point[2];
      ++x; ++y; ++z;
    }
    return msg;
  }

  void receive(Source source, const sensor_msgs::msg::PointCloud2 & msg)
  {
    receive_[source](msg);
  }

  void receive(Source source, int64_t capture_ms, std::initializer_list<XYZ> points)
  {
    receive(source, cloud(capture_ms, points));
  }

  void transform(int64_t capture_ms, double x)
  {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "map";
    msg.child_frame_id = "base_link";
    msg.header.stamp = rclcpp::Time(capture_ms * 1000000LL, RCL_ROS_TIME);
    msg.transform.translation.x = x;
    msg.transform.rotation.w = 1.0;
    ASSERT_TRUE(tf_->setTransform(msg, "async_test", false));
  }

  void update() {map_->updateMap(5.0, 5.0, 0.0);}

  unsigned char cost(double x, double y = 3.2)
  {
    unsigned int mx = 0, my = 0;
    if (!map_->getCostmap()->worldToMap(x, y, mx, my)) {
      ADD_FAILURE() << "Point outside the costmap: " << x << ", " << y;
      return NO_INFORMATION;
    }
    return map_->getCostmap()->getCost(mx, my);
  }

  void initializeObstacles()
  {
    receive(Ess, 99900, {{3.2f, 3.2f, 0.5f}});
    receive(Gemini, 99950, {{6.2f, 3.2f, 0.5f}});
    update();
    ASSERT_TRUE(map_->isCurrent());
    ASSERT_EQ(LETHAL_OBSTACLE, cost(3.2));
    ASSERT_EQ(LETHAL_OBSTACLE, cost(6.2));
  }

  nav2_util::LifecycleNode::SharedPtr node_;
  std::unique_ptr<tf2_ros::Buffer> tf_;
  std::unique_ptr<LayeredCostmap> map_;
  std::array<std::shared_ptr<Layer>, 2> layers_;
  std::array<std::function<void(const sensor_msgs::msg::PointCloud2 &)>, 2> receive_;
  std::shared_ptr<InflationLayer> inflation_;
};

TEST_P(ObstacleDecayAsyncSources, DifferentCaptureTimesUseTheirOwnTransform)
{
  transform(99900, 0.0);
  transform(99950, 2.0);
  transform(100000, 4.0);
  receive(Ess, cloud(99900, {{1.2f, 3.2f, 0.5f}}, "base_link"));
  receive(Gemini, cloud(99950, {{1.2f, 3.2f, 0.5f}}, "base_link"));
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(1.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));
  EXPECT_EQ(FREE_SPACE, cost(5.2));  // Neither cloud uses the latest robot pose.
}

TEST_P(ObstacleDecayAsyncSources, NormalEmptyInputClearsOnlyItsOwnLayer)
{
  initializeObstacles();
  receive(Ess, 99980, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));

  receive(Ess, 99990, {{4.2f, 3.2f, 0.5f}});
  receive(Gemini, 99990, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));
  EXPECT_EQ(FREE_SPACE, cost(6.2));

  receive(Ess, 100000, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(4.2));  // MAX does not retain an old master cell.
}

TEST_P(ObstacleDecayAsyncSources, MissingGeminiDoesNotBlockEssUpdatesButPreventsNavigation)
{
  receive(Ess, 99950, {{3.2f, 3.2f, 0.5f}});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));
  receive(Ess, 99990, {{4.2f, 3.2f, 0.5f}});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));

  receive(Gemini, 99995, {});
  EXPECT_FALSE(map_->isCurrent());  // Arrival alone does not refresh the applied map.
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));
}

TEST_P(ObstacleDecayAsyncSources, ExpiredGeminiRetainsObstacleWhileEssUpdates)
{
  initializeObstacles();
  setTime(100260);
  receive(Ess, 100250, {{4.2f, 3.2f, 0.5f}});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));

  receive(Gemini, 100255, {});
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));
  EXPECT_EQ(FREE_SPACE, cost(6.2));
}

TEST_P(ObstacleDecayAsyncSources, ExpiredEssStillPreventsNavigationWhileGeminiUpdates)
{
  initializeObstacles();
  setTime(100260);
  receive(Gemini, 100250, {{7.2f, 3.2f, 0.5f}});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));
  EXPECT_EQ(FREE_SPACE, cost(6.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(7.2));

  receive(Ess, 100255, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(7.2));
}

TEST_P(ObstacleDecayAsyncSources, InvalidGeminiCannotClearAndDoesNotBlockEssUpdates)
{
  initializeObstacles();
  auto malformed = cloud(99980, {});
  malformed.fields.clear();
  receive(Gemini, malformed);
  receive(Ess, 99980, {{4.2f, 3.2f, 0.5f}});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(4.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));

  receive(Gemini, cloud(99990, {}, "unconnected_camera"));
  receive(Ess, 99990, {});
  update();
  EXPECT_FALSE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(4.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));

  receive(Gemini, 100000, {});
  EXPECT_FALSE(map_->isCurrent());
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(6.2));
}

TEST_P(ObstacleDecayAsyncSources, InflationRunsAfterBothLayersAndClearsWithItsSource)
{
  configure(true);
  initializeObstacles();
  EXPECT_GT(cost(2.2), FREE_SPACE);
  EXPECT_LT(cost(2.2), LETHAL_OBSTACLE);
  EXPECT_GT(cost(7.2), FREE_SPACE);
  EXPECT_LT(cost(7.2), LETHAL_OBSTACLE);

  setTime(100260);
  receive(Ess, 100250, {});
  update();
  EXPECT_FALSE(map_->isCurrent());  // Inflation cannot hide stale Gemini health.
  EXPECT_EQ(FREE_SPACE, cost(3.2));
  EXPECT_EQ(FREE_SPACE, cost(2.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));
  EXPECT_GT(cost(7.2), FREE_SPACE);  // Retained Gemini obstacle is still inflated.

  receive(Gemini, 100255, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(6.2));
  EXPECT_EQ(FREE_SPACE, cost(7.2));
}

TEST_P(ObstacleDecayAsyncSources, RegionHistoriesAndBlindAreaRetentionRemainIndependent)
{
  if (GetParam() != DecayKind::RegionAware) {
    GTEST_SKIP() << "Blind-area retention belongs to RegionAwareDecayLayer";
  }
  std::vector<rclcpp::Parameter> parameters;
  for (const auto & name : {"ess", "gemini"}) {
    const std::string prefix = std::string(name) + ".";
    const std::vector<rclcpp::Parameter> additions{
      rclcpp::Parameter(prefix + "free_min_misses", 2),
      rclcpp::Parameter(prefix + "free_window", 3),
      rclcpp::Parameter(prefix + "blind_region_names", std::vector<std::string>{"rear"}),
      rclcpp::Parameter(prefix + "blind_region.rear.x_min", -2.0),
      rclcpp::Parameter(prefix + "blind_region.rear.x_max", -1.0),
      rclcpp::Parameter(prefix + "blind_region.rear.y_min", -2.0),
      rclcpp::Parameter(prefix + "blind_region.rear.y_max", -1.0),
      rclcpp::Parameter(prefix + "blind_region.rear.hold_seconds", 0.2)};
    parameters.insert(parameters.end(), additions.begin(), additions.end());
  }
  configure(false, parameters);
  initializeObstacles();
  receive(Ess, 99970, {});
  receive(Gemini, 99970, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));
  EXPECT_EQ(LETHAL_OBSTACLE, cost(6.2));  // One miss is insufficient in either layer.

  receive(Ess, 99980, {});
  receive(Gemini, 99980, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));  // Rear cell is still within its hold time.
  EXPECT_EQ(FREE_SPACE, cost(6.2));  // Gemini's cell is outside the blind area.

  setTime(100100);
  receive(Ess, 100090, {});
  receive(Gemini, 100090, {{7.2f, 3.2f, 0.5f}});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(LETHAL_OBSTACLE, cost(3.2));  // Receipt time must not expire the hold early.
  EXPECT_EQ(LETHAL_OBSTACLE, cost(7.2));

  receive(Ess, 100100, {});
  update();
  EXPECT_TRUE(map_->isCurrent());
  EXPECT_EQ(FREE_SPACE, cost(3.2));  // The ESS observation now reaches the hold deadline.
  EXPECT_EQ(LETHAL_OBSTACLE, cost(7.2));
}

INSTANTIATE_TEST_SUITE_P(
  BothDecayTypes, ObstacleDecayAsyncSources,
  ::testing::Values(DecayKind::Snapshot, DecayKind::RegionAware),
  [](const ::testing::TestParamInfo<DecayKind> & info) {
    return info.param == DecayKind::Snapshot ? "ObstacleDecay" : "RegionAwareDecay";
  });
}  // namespace nav2_costmap_2d
