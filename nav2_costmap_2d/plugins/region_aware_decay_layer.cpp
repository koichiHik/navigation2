// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause
/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *         Steve Macenski
 *********************************************************************/
#include "nav2_costmap_2d/region_aware_decay_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::RegionAwareDecayLayer, nav2_costmap_2d::Layer)

namespace nav2_costmap_2d
{
namespace
{
constexpr int kMaximumHistory = 1024;
constexpr int kMaximumQueue = 65536;

void require(bool condition, const std::string & reason)
{
  if (!condition) {
    throw std::invalid_argument("RegionAwareDecayLayer: " + reason);
  }
}

bool finiteNonnegative(double value)
{
  return std::isfinite(value) && value >= 0.0;
}
}  // namespace

void RegionAwareDecayLayer::History::append(bool evidence, unsigned int window)
{
  const auto value = static_cast<uint8_t>(evidence);
  if (values.size() < window) {
    values.push_back(value);
    count += value;
    return;
  }
  count -= values[next];
  values[next] = value;
  count += value;
  next = (next + 1) % window;
}

void RegionAwareDecayLayer::History::clear()
{
  values.clear();
  next = 0;
  count = 0;
}

RegionAwareDecayLayer::~RegionAwareDecayLayer()
{
  dyn_params_handler_.reset();
  for (auto & notifier : observation_notifiers_) {
    notifier.reset();
  }
}

void RegionAwareDecayLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("RegionAwareDecayLayer: failed to lock node");
  }
  clock_ = node->get_clock();
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("footprint_clearing_enabled", rclcpp::ParameterValue(true));
  declareParameter("min_obstacle_height", rclcpp::ParameterValue(0.0));
  declareParameter("max_obstacle_height", rclcpp::ParameterValue(2.0));
  declareParameter("combination_method", rclcpp::ParameterValue(1));
  declareParameter("observation_sources", rclcpp::ParameterValue(std::string("")));
  declareParameter("obstacle_pointcount", rclcpp::ParameterValue(1));
  declareParameter("occupy_min_hits", rclcpp::ParameterValue(1));
  declareParameter("occupy_window", rclcpp::ParameterValue(1));
  declareParameter("free_min_misses", rclcpp::ParameterValue(1));
  declareParameter("free_window", rclcpp::ParameterValue(1));
  declareParameter("observation_queue_size", rclcpp::ParameterValue(64));
  declareParameter("blind_region_names", rclcpp::ParameterValue(std::vector<std::string>{}));

  std::string sources;
  int occupy_hits, occupy_window, free_misses, free_window, queue_size;
  bool track_unknown;
  double transform_tolerance;
  std::vector<std::string> regions;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".footprint_clearing_enabled", footprint_clearing_enabled_);
  node->get_parameter(name_ + ".min_obstacle_height", min_obstacle_height_);
  node->get_parameter(name_ + ".max_obstacle_height", max_obstacle_height_);
  node->get_parameter(name_ + ".combination_method", combination_method_);
  node->get_parameter(name_ + ".observation_sources", sources);
  node->get_parameter(name_ + ".obstacle_pointcount", obstacle_pointcount_);
  node->get_parameter(name_ + ".occupy_min_hits", occupy_hits);
  node->get_parameter(name_ + ".occupy_window", occupy_window);
  node->get_parameter(name_ + ".free_min_misses", free_misses);
  node->get_parameter(name_ + ".free_window", free_window);
  node->get_parameter(name_ + ".observation_queue_size", queue_size);
  node->get_parameter(name_ + ".blind_region_names", regions);
  node->get_parameter("track_unknown_space", track_unknown);
  node->get_parameter("transform_tolerance", transform_tolerance);
  require(occupy_hits >= 1 && occupy_hits <= occupy_window && occupy_window <= kMaximumHistory,
    "require 1 <= occupy_min_hits <= occupy_window <= 1024");
  require(free_misses >= 1 && free_misses <= free_window && free_window <= kMaximumHistory,
    "require 1 <= free_min_misses <= free_window <= 1024");
  require(queue_size >= 1 && queue_size <= kMaximumQueue,
    "observation_queue_size must be in [1, 65536]");
  require(obstacle_pointcount_ >= 1, "obstacle_pointcount must be positive");
  require(combination_method_ == 0 || combination_method_ == 1,
    "combination_method must be 0 (Overwrite) or 1 (Maximum)");
  require(std::isfinite(min_obstacle_height_) && std::isfinite(max_obstacle_height_) &&
    min_obstacle_height_ <= max_obstacle_height_, "invalid layer height interval");
  require(finiteNonnegative(transform_tolerance), "invalid transform_tolerance");
  occupy_min_hits_ = static_cast<unsigned int>(occupy_hits);
  occupy_window_ = static_cast<unsigned int>(occupy_window);
  free_min_misses_ = static_cast<unsigned int>(free_misses);
  free_window_ = static_cast<unsigned int>(free_window);
  observation_queue_size_ = static_cast<size_t>(queue_size);

  std::set<std::string> distinct_names;
  for (const auto & region_name : regions) {
    require(!region_name.empty() && region_name.find('.') == std::string::npos &&
      distinct_names.insert(region_name).second, "blind region names must be nonempty and unique");
    const auto prefix = "blind_region." + region_name + ".";
    for (const auto * key : {"x_min", "x_max", "y_min", "y_max", "hold_seconds"}) {
      // Missing geometry must fail validation, rather than silently become a box at zero.
      declareParameter(prefix + key, rclcpp::ParameterValue(std::numeric_limits<double>::quiet_NaN()));
    }
    BlindRegion region;
    region.name = region_name;
    node->get_parameter(name_ + "." + prefix + "x_min", region.x_min);
    node->get_parameter(name_ + "." + prefix + "x_max", region.x_max);
    node->get_parameter(name_ + "." + prefix + "y_min", region.y_min);
    node->get_parameter(name_ + "." + prefix + "y_max", region.y_max);
    node->get_parameter(name_ + "." + prefix + "hold_seconds", region.hold_seconds);
    require(std::isfinite(region.x_min) && std::isfinite(region.x_max) &&
      std::isfinite(region.y_min) && std::isfinite(region.y_max) &&
      region.x_min < region.x_max && region.y_min < region.y_max,
      "invalid geometry for blind region " + region.name);
    require(std::isfinite(region.hold_seconds) &&
      (region.hold_seconds >= 0.0 || region.hold_seconds == -1.0),
      "hold_seconds must be -1 or finite and nonnegative: " + region.name);
    blind_regions_.push_back(region);
  }
  std::istringstream source_stream(sources);
  std::vector<std::string> source_names;
  std::string token;
  while (source_stream >> token) {
    source_names.push_back(token);
  }
  require(source_names.size() == 1, "exactly one PointCloud2 marking source is supported");
  source_name_ = source_names.front();
  const auto source_prefix = source_name_ + ".";
  declareParameter(source_prefix + "topic", rclcpp::ParameterValue(source_name_));
  declareParameter(source_prefix + "sensor_frame", rclcpp::ParameterValue(std::string("")));
  declareParameter(source_prefix + "observation_persistence", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "expected_update_rate", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "max_observation_age", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "data_type", rclcpp::ParameterValue(std::string("PointCloud2")));
  declareParameter(source_prefix + "min_obstacle_height", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "max_obstacle_height", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "inf_is_valid", rclcpp::ParameterValue(false));
  declareParameter(source_prefix + "marking", rclcpp::ParameterValue(true));
  declareParameter(source_prefix + "clearing", rclcpp::ParameterValue(false));
  declareParameter(source_prefix + "obstacle_min_range", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "obstacle_max_range", rclcpp::ParameterValue(2.5));
  declareParameter(source_prefix + "raytrace_min_range", rclcpp::ParameterValue(0.0));
  declareParameter(source_prefix + "raytrace_max_range", rclcpp::ParameterValue(3.0));

  std::string topic, sensor_frame, data_type;
  double persistence, expected_rate, maximum_age, source_min_height, source_max_height;
  double obstacle_min_range, obstacle_max_range, raytrace_min_range, raytrace_max_range;
  bool inf_is_valid, marking, clearing;
  node->get_parameter(name_ + "." + source_prefix + "topic", topic);
  node->get_parameter(name_ + "." + source_prefix + "sensor_frame", sensor_frame);
  node->get_parameter(name_ + "." + source_prefix + "observation_persistence", persistence);
  node->get_parameter(name_ + "." + source_prefix + "expected_update_rate", expected_rate);
  node->get_parameter(name_ + "." + source_prefix + "max_observation_age", maximum_age);
  node->get_parameter(name_ + "." + source_prefix + "data_type", data_type);
  node->get_parameter(name_ + "." + source_prefix + "min_obstacle_height", source_min_height);
  node->get_parameter(name_ + "." + source_prefix + "max_obstacle_height", source_max_height);
  node->get_parameter(name_ + "." + source_prefix + "inf_is_valid", inf_is_valid);
  node->get_parameter(name_ + "." + source_prefix + "marking", marking);
  node->get_parameter(name_ + "." + source_prefix + "clearing", clearing);
  node->get_parameter(name_ + "." + source_prefix + "obstacle_min_range", obstacle_min_range);
  node->get_parameter(name_ + "." + source_prefix + "obstacle_max_range", obstacle_max_range);
  node->get_parameter(name_ + "." + source_prefix + "raytrace_min_range", raytrace_min_range);
  node->get_parameter(name_ + "." + source_prefix + "raytrace_max_range", raytrace_max_range);
  require(!topic.empty(), "source topic must not be empty");
  require(data_type == "PointCloud2", "only PointCloud2 input is supported");
  require(persistence == 0.0, "observation_persistence must be 0 (queue owns observation history)");
  require(marking && !clearing,
    "source must use marking=true and clearing=false");
  require(!inf_is_valid, "inf_is_valid is unsupported for PointCloud2");
  require(std::isfinite(expected_rate) && expected_rate > 0.0 &&
    std::isfinite(maximum_age) && maximum_age > 0.0,
    "positive expected_update_rate and max_observation_age are required for fail-safe freshness");
  require(std::isfinite(source_min_height) && std::isfinite(source_max_height) &&
    source_min_height <= source_max_height, "invalid source height interval");
  require(finiteNonnegative(obstacle_min_range) && std::isfinite(obstacle_max_range) &&
    obstacle_max_range > obstacle_min_range, "invalid obstacle range interval");
  require(finiteNonnegative(raytrace_min_range) && std::isfinite(raytrace_max_range) &&
    raytrace_max_range >= raytrace_min_range, "invalid raytrace range interval");

  rolling_window_ = layered_costmap_->isRolling();
  global_frame_ = layered_costmap_->getGlobalFrameID();
  default_value_ = track_unknown ? NO_INFORMATION : FREE_SPACE;
  matchSize();
  observation_buffers_.push_back(std::make_shared<ObservationBufferWithBase>(
      node, topic, persistence, expected_rate, source_min_height, source_max_height,
      obstacle_max_range, obstacle_min_range, raytrace_max_range, raytrace_min_range,
      *tf_, global_frame_, sensor_frame, tf2::durationFromSec(transform_tolerance), maximum_age));
  auto options = rclcpp::SubscriptionOptions();
  options.callback_group = callback_group_;
  auto qos = rmw_qos_profile_sensor_data;
  qos.depth = 50;
  auto subscriber = std::make_shared<message_filters::Subscriber<
      sensor_msgs::msg::PointCloud2, rclcpp_lifecycle::LifecycleNode>>(node, topic, qos, options);
  subscriber->unsubscribe();
  auto filter = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
    *subscriber, *tf_, global_frame_, 50, node->get_node_logging_interface(),
    node->get_node_clock_interface(), tf2::durationFromSec(transform_tolerance));
  filter->registerCallback(std::bind(&RegionAwareDecayLayer::pointCloud2Callback, this,
      std::placeholders::_1, observation_buffers_.front()));
  if (!sensor_frame.empty()) {
    filter->setTargetFrames(std::vector<std::string>{global_frame_, sensor_frame});
  }
  observation_subscribers_.push_back(subscriber);
  observation_notifiers_.push_back(filter);
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(&RegionAwareDecayLayer::dynamicParametersCallback, this, std::placeholders::_1));
  current_ = false;
  RCLCPP_INFO(logger_,
    "RegionAwareDecayLayer source=%s queue=%zu a=%u/%u b=%u/%u regions=%zu",
    topic.c_str(), observation_queue_size_, occupy_min_hits_, occupy_window_,
    free_min_misses_, free_window_, blind_regions_.size());
}

rcl_interfaces::msg::SetParametersResult RegionAwareDecayLayer::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  auto node = node_.lock();
  bool requested_enabled = enabled_;
  for (const auto & parameter : parameters) {
    const auto prefix = name_ + ".";
    if (parameter.get_name().compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    if (parameter.get_name() == prefix + "enabled") {
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
        result.successful = false;
        result.reason = "enabled must be bool";
        return result;
      }
      requested_enabled = parameter.as_bool();
    } else if (!node || !node->has_parameter(parameter.get_name()) ||
      node->get_parameter(parameter.get_name()).get_parameter_value() != parameter.get_parameter_value())
    {
      result.successful = false;
      result.reason = "RegionAwareDecayLayer configuration is read-only while running; reconfigure to change " +
        parameter.get_name();
      return result;
    }
  }
  if (requested_enabled != enabled_) {
    enabled_ = requested_enabled;
    clearInput();
    clearHistories(false);
    Costmap2D::resetMaps();
  }
  return result;
}

void RegionAwareDecayLayer::clearInput()
{
  queue_.clear();
  latest_snapshot_.reset();
  pending_observations_.clear();
  applied_observations_.clear();
  update_pending_ = false;
  current_ = false;
  last_enqueued_capture_ns_ = 0;
  last_enqueued_generation_ = 0;
  for (auto & notifier : observation_notifiers_) {
    notifier->clear();
  }
  for (auto & buffer : observation_buffers_) {
    buffer->resetLastUpdated();
  }
}

void RegionAwareDecayLayer::filterClearedObservations(
  int start_x, int start_y, int end_x, int end_y, bool invert)
{
  bool latest_filtered = false;
  for (auto & event : queue_) {
    const bool is_latest = event == latest_snapshot_;
    event = filterClearedObservation(event, start_x, start_y, end_x, end_y, invert);
    if (is_latest) {
      latest_snapshot_ = event;
      latest_filtered = true;
    }
  }
  if (!latest_filtered) {
    latest_snapshot_ = filterClearedObservation(
      latest_snapshot_, start_x, start_y, end_x, end_y, invert);
  }
}

std::shared_ptr<const RegionAwareDecayLayer::QueuedObservation>
RegionAwareDecayLayer::filterClearedObservation(
  const std::shared_ptr<const QueuedObservation> & event,
  int start_x, int start_y, int end_x, int end_y, bool invert) const
{
  if (!event || event->observation->cloud_->data.empty()) {
    return event;
  }
  // Copy the cloud without changing the source buffer or snapshot freshness.
  auto observation = std::make_shared<Observation>(*event->observation);
  removeClearedPoints(*observation->cloud_, start_x, start_y, end_x, end_y, invert);
  auto filtered = std::make_shared<QueuedObservation>(*event);
  filtered->observation = std::move(observation);
  return filtered;
}

void RegionAwareDecayLayer::removeClearedPoints(
  sensor_msgs::msg::PointCloud2 & cloud,
  int start_x, int start_y, int end_x, int end_y, bool invert) const
{
  sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y");
  size_t read_offset = 0;
  size_t write_offset = 0;
  for (; x != x.end(); ++x, ++y, read_offset += cloud.point_step) {
    unsigned int mx, my;
    if (worldToMap(*x, *y, mx, my)) {
      const bool inside = static_cast<int>(mx) > start_x && static_cast<int>(mx) < end_x &&
        static_cast<int>(my) > start_y && static_cast<int>(my) < end_y;
      if (inside != invert) {
        continue;
      }
    }
    // Keep complete point records, including fields other than xyz.
    if (write_offset != read_offset) {
      std::memmove(
        cloud.data.data() + write_offset, cloud.data.data() + read_offset, cloud.point_step);
    }
    write_offset += cloud.point_step;
  }
  if (write_offset != cloud.data.size()) {
    cloud.height = 1;
    sensor_msgs::PointCloud2Modifier(cloud).resize(write_offset / cloud.point_step);
  }
}

void RegionAwareDecayLayer::clearHistories(bool retain_occupied)
{
  for (auto & cell : cells_) {
    cell.a.clear();
    cell.b.clear();
    if (!retain_occupied) {
      cell.occupied = false;
      cell.latest_hit = false;
      cell.last_seen_ns = 0;
    }
  }
}

bool RegionAwareDecayLayer::checkClock()
{
  const auto now = clock_->now().nanoseconds();
  const bool backwards = last_clock_ns_ >= 0 && now < last_clock_ns_;
  last_clock_ns_ = now;
  if (backwards) {
    ++clock_resets_;
    clearInput();
    clearHistories(false);
    Costmap2D::resetMaps();
    RCLCPP_WARN(logger_, "RegionAwareDecayLayer ROS time moved backwards; history and input reset");
  }
  return backwards;
}

void RegionAwareDecayLayer::pointCloud2Callback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr message,
  const std::shared_ptr<ObservationBufferWithBase> & buffer)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!active_ || !enabled_ || !message || !buffer) {
    return;
  }
  checkClock();
  buffer->lock();
  buffer->bufferCloud(*message);
  std::vector<Observation> observations;
  ObservationBufferWithBase::Freshness freshness;
  buffer->getObservations(observations, &freshness);
  const bool valid = buffer->isCurrent(freshness);
  buffer->unlock();
  if (!valid || observations.empty()) {
    // A failed callback revokes all pre-error queued snapshots as well as the
    // applied generation. Retain the grid and cell state while input is unavailable.
    queue_.clear();
    latest_snapshot_.reset();
    pending_observations_.clear();
    update_pending_ = false;
    current_ = false;
    RCLCPP_WARN(logger_, "RegionAwareDecayLayer rejected observation; retained grid is unavailable");
    return;
  }
  if (freshness.capture_ns == last_enqueued_capture_ns_ &&
    freshness.generation == last_enqueued_generation_)
  {
    return;
  }
  if (queue_.size() >= observation_queue_size_) {
    ++queue_overflows_;
    queue_.clear();
    latest_snapshot_.reset();
    pending_observations_.clear();
    update_pending_ = false;
    current_ = false;
    buffer->invalidate();
    RCLCPP_ERROR(logger_,
      "RegionAwareDecayLayer observation queue overflow (%zu); retained grid unavailable until new input",
      observation_queue_size_);
    return;
  }
  auto event = std::make_shared<QueuedObservation>();
  event->observation = std::make_shared<Observation>(observations.front());
  event->freshness = BufferedFreshness{buffer, freshness};
  event->source = source_name_;
  queue_.push_back(event);
  latest_snapshot_ = event;
  last_enqueued_capture_ns_ = freshness.capture_ns;
  last_enqueued_generation_ = freshness.generation;
  queue_high_watermark_ = std::max(queue_high_watermark_, queue_.size());
  RCLCPP_DEBUG(logger_, "RegionAwareDecayLayer queued source=%s capture_ns=%ld received_ns=%ld size=%zu",
    source_name_.c_str(), freshness.capture_ns, freshness.received_ns, queue_.size());
}

void RegionAwareDecayLayer::matchSize()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  CostmapLayer::matchSize();
  cells_.clear();
  cells_.resize(static_cast<size_t>(size_x_) * size_y_);
  point_counts_.assign(cells_.size(), 0);
  clearInput();
}

void RegionAwareDecayLayer::updateOrigin(double new_origin_x, double new_origin_y)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!std::isfinite(new_origin_x) || !std::isfinite(new_origin_y) || resolution_ <= 0.0) {
    return;
  }
  // Match Costmap2D's integer truncation and snapped origin exactly. Never
  // override resetMaps(): Costmap2D calls it internally while moving the grid.
  const int dx = static_cast<int>((new_origin_x - origin_x_) / resolution_);
  const int dy = static_cast<int>((new_origin_y - origin_y_) / resolution_);
  if (dx == 0 && dy == 0) {
    return;
  }
  std::vector<CellHistory> shifted(cells_.size());
  for (int y = 0; y < static_cast<int>(size_y_); ++y) {
    const int previous_y = y + dy;
    if (previous_y < 0 || previous_y >= static_cast<int>(size_y_)) {
      continue;
    }
    for (int x = 0; x < static_cast<int>(size_x_); ++x) {
      const int previous_x = x + dx;
      if (previous_x >= 0 && previous_x < static_cast<int>(size_x_)) {
        shifted[getIndex(x, y)] = std::move(cells_[getIndex(previous_x, previous_y)]);
      }
    }
  }
  Costmap2D::updateOrigin(new_origin_x, new_origin_y);
  cells_.swap(shifted);
}

void RegionAwareDecayLayer::clearArea(int start_x, int start_y, int end_x, int end_y, bool invert)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  CostmapLayer::clearArea(start_x, start_y, end_x, end_y, invert);
  for (int y = 0; y < static_cast<int>(size_y_); ++y) {
    for (int x = 0; x < static_cast<int>(size_x_); ++x) {
      const bool inside = x > start_x && x < end_x && y > start_y && y < end_y;
      if (inside != invert) {
        cells_[getIndex(x, y)] = CellHistory{};
      }
    }
  }
  // Remove only cleared cells from pre-clear snapshots; keep outside evidence
  // and empty observations so their original event timeline remains intact.
  filterClearedObservations(start_x, start_y, end_x, end_y, invert);
  addExtraBounds(origin_x_, origin_y_, origin_x_ + getSizeInMetersX(), origin_y_ + getSizeInMetersY());
}

void RegionAwareDecayLayer::createHitMask(const Observation & observation)
{
  point_counts_.assign(cells_.size(), 0);
  const auto & cloud = *observation.cloud_;
  const double maximum_distance = observation.obstacle_max_range_ * observation.obstacle_max_range_;
  const double minimum_distance = observation.obstacle_min_range_ * observation.obstacle_min_range_;
  sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
  for (; x != x.end(); ++x, ++y, ++z) {
    const double height = *z - observation.origin_.z;
    if (height < min_obstacle_height_ || height > max_obstacle_height_) {
      continue;
    }
    const double dx = *x - observation.origin_.x;
    const double dy = *y - observation.origin_.y;
    const double dz = *z - observation.origin_.z;
    const double distance = dx * dx + dy * dy + dz * dz;
    if (distance >= maximum_distance || distance < minimum_distance) {
      continue;
    }
    unsigned int mx, my;
    if (worldToMap(*x, *y, mx, my)) {
      auto & count = point_counts_[getIndex(mx, my)];
      if (count < static_cast<unsigned int>(obstacle_pointcount_)) {
        ++count;
      }
    }
  }
}

double RegionAwareDecayLayer::holdingTime(
  size_t index, double robot_x, double robot_y, double robot_yaw, std::string * region_name) const
{
  double wx, wy;
  mapToWorld(static_cast<unsigned int>(index % size_x_), static_cast<unsigned int>(index / size_x_), wx, wy);
  const double c = std::cos(robot_yaw), s = std::sin(robot_yaw);
  const double x = c * (wx - robot_x) + s * (wy - robot_y);
  const double y = -s * (wx - robot_x) + c * (wy - robot_y);
  double hold = 0.0;
  if (region_name) {
    *region_name = "none";
  }
  for (const auto & region : blind_regions_) {
    if (x < region.x_min || x > region.x_max || y < region.y_min || y > region.y_max) {
      continue;
    }
    if (region.hold_seconds == -1.0 || region.hold_seconds >= hold) {
      hold = region.hold_seconds;
      if (region_name) {
        *region_name = region.name;
      }
    }
    if (hold == -1.0) {
      break;
    }
  }
  return hold;
}

bool RegionAwareDecayLayer::mayFree(
  size_t index, double robot_x, double robot_y, double robot_yaw, int64_t observation_ns) const
{
  const auto & cell = cells_[index];
  if (cell.latest_hit || cell.b.count < free_min_misses_) {
    return false;
  }
  std::string region;
  const double hold = holdingTime(index, robot_x, robot_y, robot_yaw, &region);
  const bool may_free = hold != -1.0 && (hold == 0.0 ||
    (observation_ns >= cell.last_seen_ns &&
    static_cast<double>(observation_ns - cell.last_seen_ns) / 1e9 >= hold));
  if (!may_free) {
    RCLCPP_DEBUG(logger_,
      "RegionAwareDecayLayer retained index=%zu region=%s last_seen_ns=%ld hold_seconds=%.3f",
      index, region.c_str(), cell.last_seen_ns, hold);
  }
  return may_free;
}

void RegionAwareDecayLayer::applyObservation(
  const QueuedObservation & event, double robot_x, double robot_y,
  double robot_yaw, int64_t observation_ns)
{
  createHitMask(*event.observation);
  for (size_t index = 0; index < cells_.size(); ++index) {
    auto & cell = cells_[index];
    const bool hit = point_counts_[index] >= static_cast<unsigned int>(obstacle_pointcount_);
    cell.latest_hit = hit;
    if (hit) {
      cell.last_seen_ns = event.freshness.observation.capture_ns;
    }
    if (!cell.occupied) {
      cell.a.append(hit, occupy_window_);
      if (cell.a.count >= occupy_min_hits_) {
        cell.occupied = true;
        cell.b.clear();
        RCLCPP_DEBUG(logger_, "RegionAwareDecayLayer OCCUPIED index=%zu capture_ns=%ld",
          index, cell.last_seen_ns);
      }
    } else {
      cell.b.append(!hit, free_window_);
      if (!hit && mayFree(index, robot_x, robot_y, robot_yaw, observation_ns)) {
        cell.occupied = false;
        cell.a.clear();
        RCLCPP_DEBUG(logger_, "RegionAwareDecayLayer FREE index=%zu last_seen_ns=%ld",
          index, cell.last_seen_ns);
      }
    }
  }
}

void RegionAwareDecayLayer::rebuildGrid()
{
  for (size_t index = 0; index < cells_.size(); ++index) {
    costmap_[index] = cells_[index].occupied ? LETHAL_OBSTACLE : FREE_SPACE;
  }
}

void RegionAwareDecayLayer::updateFootprint(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!footprint_clearing_enabled_) {
    return;
  }
  transformFootprint(robot_x, robot_y, robot_yaw, getFootprint(), transformed_footprint_);
  for (const auto & point : transformed_footprint_) {
    touch(point.x, point.y, min_x, min_y, max_x, max_y);
  }
}

void RegionAwareDecayLayer::clearFootprintHistory()
{
  std::vector<MapLocation> polygon;
  for (const auto & point : transformed_footprint_) {
    MapLocation location;
    if (!worldToMap(point.x, point.y, location.x, location.y)) {
      // Match setConvexPolygonCost's all-or-nothing boundary behavior.
      return;
    }
    polygon.push_back(location);
  }
  std::vector<MapLocation> polygon_cells;
  convexFillCells(polygon, polygon_cells);
  for (const auto & location : polygon_cells) {
    const auto index = getIndex(location.x, location.y);
    cells_[index] = CellHistory{};
    costmap_[index] = FREE_SPACE;
  }
}

void RegionAwareDecayLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (rolling_window_) {
    updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
  }
  if (!enabled_) {
    return;
  }
  checkClock();
  useExtraBounds(min_x, min_y, max_x, max_y);
  // Match the old layer's full retained-grid merge even when input is missing.
  *min_x = robot_x - getSizeInMetersX();
  *min_y = robot_y - getSizeInMetersY();
  *max_x = robot_x + getSizeInMetersX();
  *max_y = robot_y + getSizeInMetersY();
  current_ = false;
  update_pending_ = false;
  pending_observations_.clear();
  size_t processed = 0;
  const auto finish = [this, &started, &processed]() {
      pending_processed_ = processed;
      bounds_update_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      if (!update_pending_) {
        logDiagnostics(processed, bounds_update_ms_);
      }
    };
  if (!active_ || !latest_snapshot_ ||
    !latest_snapshot_->freshness.buffer->isCurrent(latest_snapshot_->freshness.observation))
  {
    stale_dropped_ += queue_.size();
    queue_.clear();
    finish();
    return;
  }
  std::stable_sort(queue_.begin(), queue_.end(), [](const auto & left, const auto & right) {
      return left->freshness.observation.capture_ns < right->freshness.observation.capture_ns;
    });
  while (!queue_.empty()) {
    auto event = queue_.front();
    queue_.pop_front();
    if (!event->freshness.buffer->isCurrent(event->freshness.observation)) {
      ++stale_dropped_;
      RCLCPP_WARN(logger_, "RegionAwareDecayLayer expired queued observation capture_ns=%ld",
        event->freshness.observation.capture_ns);
      continue;
    }
    // Use each event's capture stamp for retention as well as history updates.
    // A delayed, pre-expiry miss must not free a cell before a later queued hit.
    applyObservation(*event, robot_x, robot_y, robot_yaw, event->freshness.observation.capture_ns);
    pending_observations_ = {event->freshness};
    ++processed;
    ++observations_processed_;
  }
  if (processed == 0) {
    if (applied_observations_.empty() || !applied_observations_.front().buffer->isCurrent(
        applied_observations_.front().observation))
    {
      finish();
      return;
    }
    pending_observations_ = applied_observations_;
  }
  rebuildGrid();
  updateFootprint(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);
  update_pending_ = true;
  finish();
}

void RegionAwareDecayLayer::updateCosts(
  Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j)
{
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_) {
    return;
  }
  if (update_pending_ && footprint_clearing_enabled_) {
    // Exactly the same rasterization as the existing polygon clearing; clear
    // both histories for those raster cells so retained obstacles cannot revive.
    setConvexPolygonCost(transformed_footprint_, FREE_SPACE);
    clearFootprintHistory();
  }
  if (combination_method_ == 0) {
    updateWithOverwrite(master_grid, min_i, min_j, max_i, max_j);
  } else {
    updateWithMax(master_grid, min_i, min_j, max_i, max_j);
  }
  if (update_pending_) {
    applied_observations_ = pending_observations_;
    update_pending_ = false;
    current_ = true;
    logDiagnostics(pending_processed_, bounds_update_ms_ +
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
  }
}

bool RegionAwareDecayLayer::isCurrent() const
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!current_ || applied_observations_.empty()) {
    return false;
  }
  for (const auto & snapshot : applied_observations_) {
    if (!snapshot.buffer->isCurrent(snapshot.observation)) {
      return false;
    }
  }
  return true;
}

void RegionAwareDecayLayer::activate()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  clearInput();
  clearHistories(true);
  last_clock_ns_ = clock_->now().nanoseconds();
  active_ = true;
  for (auto & subscriber : observation_subscribers_) {
    subscriber->subscribe();
  }
}

void RegionAwareDecayLayer::deactivate()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  active_ = false;
  for (auto & subscriber : observation_subscribers_) {
    subscriber->unsubscribe();
  }
  clearInput();
  clearHistories(true);
}

void RegionAwareDecayLayer::resetBuffersLastUpdated()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  clearInput();
}

void RegionAwareDecayLayer::reset()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  clearInput();
  clearHistories(false);
  Costmap2D::resetMaps();
  last_clock_ns_ = clock_->now().nanoseconds();
}

void RegionAwareDecayLayer::logDiagnostics(size_t processed, double elapsed_ms)
{
  maximum_update_ms_ = std::max(maximum_update_ms_, elapsed_ms);
  size_t occupied = 0, held = 0;
  int64_t oldest_seen = 0;
  for (const auto & cell : cells_) {
    if (cell.occupied) {
      ++occupied;
      held += !cell.latest_hit;
      if (oldest_seen == 0 || cell.last_seen_ns < oldest_seen) {
        oldest_seen = cell.last_seen_ns;
      }
    }
  }
  RCLCPP_INFO_THROTTLE(logger_, *clock_, 5000,
    "RegionAwareDecayLayer processed=%zu total=%lu queue=%zu high_water=%zu overflow=%lu stale_dropped=%lu "
    "occupied=%zu held=%zu oldest_seen_ns=%ld applied_capture_ns=%ld fresh=%s update_ms=%.3f max_ms=%.3f clock_resets=%lu",
    processed, observations_processed_, queue_.size(), queue_high_watermark_, queue_overflows_, stale_dropped_,
    occupied, held, oldest_seen, applied_observations_.empty() ? 0L : applied_observations_.front().observation.capture_ns,
    isCurrent() ? "true" : "false", elapsed_ms, maximum_update_ms_, clock_resets_);
}

}  // namespace nav2_costmap_2d
