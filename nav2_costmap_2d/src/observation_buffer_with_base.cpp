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
 *********************************************************************/
#include "nav2_costmap_2d/observation_buffer_with_base.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>

#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/convert.h"
using namespace std::chrono_literals;

namespace nav2_costmap_2d {
ObservationBufferWithBase::ObservationBufferWithBase(
    const nav2_util::LifecycleNode::WeakPtr& parent, std::string topic_name,
    double observation_keep_time, double expected_update_rate,
    double min_obstacle_height, double max_obstacle_height,
    double obstacle_max_range, double obstacle_min_range,
    double raytrace_max_range, double raytrace_min_range,
    tf2_ros::Buffer& tf2_buffer, std::string global_frame,
    std::string sensor_frame, tf2::Duration tf_tolerance, double max_observation_age)
    : tf2_buffer_(tf2_buffer),
      observation_keep_time_(
          rclcpp::Duration::from_seconds(observation_keep_time)),
      expected_update_rate_(
          rclcpp::Duration::from_seconds(expected_update_rate)),
      max_observation_age_(rclcpp::Duration::from_seconds(max_observation_age)),
      global_frame_(global_frame),
      sensor_frame_(sensor_frame),
      topic_name_(topic_name),
      min_obstacle_height_(min_obstacle_height),
      max_obstacle_height_(max_obstacle_height),
      obstacle_max_range_(obstacle_max_range),
      obstacle_min_range_(obstacle_min_range),
      raytrace_max_range_(raytrace_max_range),
      raytrace_min_range_(raytrace_min_range),
      tf_tolerance_(tf_tolerance) {
  auto node = parent.lock();
  clock_ = node->get_clock();
  logger_ = node->get_logger();
  if (!std::isfinite(max_observation_age) || max_observation_age < 0.0) {
    throw std::invalid_argument("max_observation_age must be finite and nonnegative");
  }
}

ObservationBufferWithBase::~ObservationBufferWithBase() {}

void ObservationBufferWithBase::bufferCloud(
    const sensor_msgs::msg::PointCloud2& cloud) {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  geometry_msgs::msg::PointStamped global_origin;
  const auto received = clock_->now().nanoseconds();
  const int64_t stamp = static_cast<int64_t>(cloud.header.stamp.sec) * 1000000000LL +
      cloud.header.stamp.nanosec;
  if (max_observation_age_.nanoseconds() > 0 &&
      (stamp <= 0 || cloud.header.stamp.nanosec >= 1000000000u || stamp > received ||
       received - stamp >= max_observation_age_.nanoseconds() ||
       (latest_.valid && stamp <= latest_.capture_ns))) {
    invalidate();
    return;
  }

  // Commit only after validation and transforms succeed, including an empty cloud.
  Observation candidate;

  // check whether the origin frame has been set explicitly
  // or whether we should get it from the cloud
  std::string origin_frame =
      sensor_frame_ == "" ? cloud.header.frame_id : sensor_frame_;

  try {
    if (cloud.header.frame_id.empty() || cloud.height == 0 || cloud.point_step == 0 ||
        static_cast<uint64_t>(cloud.width) * cloud.point_step != cloud.row_step ||
        static_cast<uint64_t>(cloud.row_step) * cloud.height != cloud.data.size()) {
      throw std::runtime_error("invalid PointCloud2 layout");
    }
    for (const auto * field_name : {"x", "y", "z"}) {
      const auto field = std::find_if(cloud.fields.begin(), cloud.fields.end(),
          [field_name](const sensor_msgs::msg::PointField & field) {
            return field.name == field_name;
          });
      if (field == cloud.fields.end() || field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
          field->count != 1 || static_cast<uint64_t>(field->offset) + sizeof(float) > cloud.point_step) {
        throw std::runtime_error("PointCloud2 requires valid FLOAT32 x/y/z fields");
      }
    }
    // given these observations come from sensors...
    // we'll need to store the origin pt of the sensor
    geometry_msgs::msg::PointStamped local_origin;
    local_origin.header.stamp = cloud.header.stamp;
    local_origin.header.frame_id = origin_frame;
    local_origin.point.x = 0;
    local_origin.point.y = 0;
    local_origin.point.z = 0;
    tf2_buffer_.transform(local_origin, global_origin, global_frame_,
                          tf_tolerance_);
    tf2::convert(global_origin.point, candidate.origin_);

    // make sure to pass on the raytrace/obstacle range
    // of the observation buffer to the observations
    candidate.raytrace_max_range_ = raytrace_max_range_;
    candidate.raytrace_min_range_ = raytrace_min_range_;
    candidate.obstacle_max_range_ = obstacle_max_range_;
    candidate.obstacle_min_range_ = obstacle_min_range_;

    sensor_msgs::msg::PointCloud2 global_frame_cloud;

    // transform the point cloud
    tf2_buffer_.transform(cloud, global_frame_cloud, global_frame_,
                          tf_tolerance_);
    global_frame_cloud.header.stamp = cloud.header.stamp;

    // now we need to remove observations from the cloud that are below
    // or above our height thresholds
    sensor_msgs::msg::PointCloud2& observation_cloud =
        *(candidate.cloud_);
    observation_cloud.height = global_frame_cloud.height;
    observation_cloud.width = global_frame_cloud.width;
    observation_cloud.fields = global_frame_cloud.fields;
    observation_cloud.is_bigendian = global_frame_cloud.is_bigendian;
    observation_cloud.point_step = global_frame_cloud.point_step;
    observation_cloud.row_step = global_frame_cloud.row_step;
    observation_cloud.is_dense = global_frame_cloud.is_dense;

    unsigned int cloud_size =
        global_frame_cloud.height * global_frame_cloud.width;
    sensor_msgs::PointCloud2Modifier modifier(observation_cloud);
    modifier.resize(cloud_size);
    unsigned int point_count = 0;

    // copy over the points that are within our height bounds
    sensor_msgs::PointCloud2Iterator<float> iter_x(global_frame_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(global_frame_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(global_frame_cloud, "z");
    std::vector<unsigned char>::const_iterator
        iter_global = global_frame_cloud.data.begin(),
        iter_global_end = global_frame_cloud.data.end();
    std::vector<unsigned char>::iterator iter_obs =
        observation_cloud.data.begin();
    for (; iter_global != iter_global_end;
         ++iter_x, ++iter_y, ++iter_z, iter_global += global_frame_cloud.point_step) {
      if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
        throw std::runtime_error("non-finite transformed PointCloud2 coordinates");
      }
      if ((*iter_z) - global_origin.point.z <= max_obstacle_height_ &&
          min_obstacle_height_ <= (*iter_z) - global_origin.point.z) {
        std::copy(iter_global, iter_global + global_frame_cloud.point_step,
                  iter_obs);
        iter_obs += global_frame_cloud.point_step;
        ++point_count;
      }
    }

    // resize the cloud for the number of legal points
    modifier.resize(point_count);
    observation_cloud.header.stamp = cloud.header.stamp;
    observation_cloud.header.frame_id = global_frame_cloud.header.frame_id;
  } catch (const std::exception & ex) {
    invalidate();
    RCLCPP_WARN(logger_, "Cannot buffer observation from %s: %s",
        topic_name_.c_str(), ex.what());
    return;
  }

  // Transform processing time does not refresh either the capture or receipt stamp.
  observation_list_.push_front(candidate);
  latest_ = Freshness{stamp, received, true, generation_};
  input_valid_ = true;

  // we'll also remove any stale observations from the list
  purgeStaleObservations();
}

// returns a copy of the observations
void ObservationBufferWithBase::getObservations(
    std::vector<Observation>& observations, Freshness * freshness) {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  if (freshness) {
    *freshness = latest_;
  }
  // first... let's make sure that we don't have any stale observations
  purgeStaleObservations();

  // now we'll just copy the observations for the caller
  std::list<Observation>::iterator obs_it;
  for (obs_it = observation_list_.begin(); obs_it != observation_list_.end();
       ++obs_it) {
    observations.push_back(*obs_it);
  }
}

void ObservationBufferWithBase::purgeStaleObservations() {
  if (!observation_list_.empty()) {
    std::list<Observation>::iterator obs_it = observation_list_.begin();
    // if we're keeping observations for no time... then we'll only keep one
    // observation
    if (observation_keep_time_ == rclcpp::Duration(0.0s)) {
      observation_list_.erase(++obs_it, observation_list_.end());
      return;
    }

    // otherwise... we'll have to loop through the observations to see which
    // ones are stale
    for (obs_it = observation_list_.begin(); obs_it != observation_list_.end();
         ++obs_it) {
      Observation& obs = *obs_it;
      // check if the observation is out of date... and if it is,
      // remove it and those that follow from the list
      if ((clock_->now() - obs.cloud_->header.stamp) > observation_keep_time_) {
        observation_list_.erase(obs_it, observation_list_.end());
        return;
      }
    }
  }
}

bool ObservationBufferWithBase::isCurrent() const {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  return isCurrent(latest_) && !observation_list_.empty();
}

bool ObservationBufferWithBase::isCurrent(const Freshness & applied) const {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  if (!input_valid_ || !applied.valid || applied.generation != generation_) {
    return false;
  }
  const auto now = clock_->now().nanoseconds();
  if (now < applied.received_ns ||
      (expected_update_rate_.nanoseconds() > 0 &&
       now - applied.received_ns >= expected_update_rate_.nanoseconds())) {
    return false;
  }
  return max_observation_age_.nanoseconds() == 0 ||
      (now >= applied.capture_ns &&
       now - applied.capture_ns < max_observation_age_.nanoseconds());
}

void ObservationBufferWithBase::invalidate() {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  input_valid_ = false;
  ++generation_;
}

void ObservationBufferWithBase::resetLastUpdated() {
  std::lock_guard<std::recursive_mutex> guard(lock_);
  observation_list_.clear();
  latest_ = Freshness{};
  invalidate();
}
}  // namespace nav2_costmap_2d
