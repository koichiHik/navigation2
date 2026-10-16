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
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nav2_costmap_2d/costmap_math.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

PLUGINLIB_EXPORT_CLASS(nav2_costmap_2d::ObstacleDecayLayer, nav2_costmap_2d::Layer)

using nav2_costmap_2d::FREE_SPACE;
using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

using nav2_costmap_2d::Observation;
using nav2_costmap_2d::ObservationBufferWithBase;
using rcl_interfaces::msg::ParameterType;

namespace nav2_costmap_2d
{

ObstacleDecayLayer::~ObstacleDecayLayer()
{
  dyn_params_handler_.reset();
  for (auto & notifier : observation_notifiers_) {
    notifier.reset();
  }
}

void ObstacleDecayLayer::onInitialize()
{
  bool track_unknown_space;
  double transform_tolerance;

  // The topics that we'll subscribe to from the parameter server
  std::string topics_string;

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("footprint_clearing_enabled", rclcpp::ParameterValue(true));
  declareParameter("min_obstacle_height", rclcpp::ParameterValue(0.0));
  declareParameter("max_obstacle_height", rclcpp::ParameterValue(2.0));
  declareParameter("combination_method", rclcpp::ParameterValue(1));
  declareParameter("observation_sources", rclcpp::ParameterValue(std::string("")));
  declareParameter("obstacle_pointcount", rclcpp::ParameterValue(1));

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "footprint_clearing_enabled", footprint_clearing_enabled_);
  node->get_parameter(name_ + "." + "min_obstacle_height", min_obstacle_height_);
  node->get_parameter(name_ + "." + "max_obstacle_height", max_obstacle_height_);
  node->get_parameter(name_ + "." + "combination_method", combination_method_);
  node->get_parameter("track_unknown_space", track_unknown_space);
  node->get_parameter("transform_tolerance", transform_tolerance);
  node->get_parameter(name_ + "." + "observation_sources", topics_string);
  node->get_parameter(name_ + "." + "obstacle_pointcount", obstacle_pointcount_);

  if (!std::isfinite(min_obstacle_height_) || !std::isfinite(max_obstacle_height_) ||
    min_obstacle_height_ > max_obstacle_height_)
  {
    throw std::invalid_argument(
            "obstacle height limits must be finite and min_obstacle_height must not exceed "
            "max_obstacle_height");
  }
  if (combination_method_ != 0 && combination_method_ != 1) {
    throw std::invalid_argument("combination_method must be 0 (overwrite) or 1 (maximum)");
  }

  RCLCPP_INFO(logger_, "Subscribed to Topics: %s", topics_string.c_str());

  rolling_window_ = layered_costmap_->isRolling();

  if (track_unknown_space) {
    default_value_ = NO_INFORMATION;
  } else {
    default_value_ = FREE_SPACE;
  }

  ObstacleDecayLayer::matchSize();
  current_ = false;

  global_frame_ = layered_costmap_->getGlobalFrameID();

  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group_;

  const bool snapshot_input_initialized = initializeSnapshotInput(topics_string);
  // Register dynamic callbacks only after every startup-only snapshot parameter
  // has consumed its launch override. Otherwise parameter declaration itself
  // would be mistaken for an unsafe runtime policy change.
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(&ObstacleDecayLayer::dynamicParametersCallback, this, std::placeholders::_1));
  if (snapshot_input_initialized) {
    default_value_ = unreported_cells_are_free_ ? FREE_SPACE : NO_INFORMATION;
    resetMaps();
    return;
  }

  // now we need to split the topics based on whitespace which we can use a
  // stringstream for
  std::stringstream ss(topics_string);

  std::string source;
  while (ss >> source) {
    // get the parameters for the specific topic
    double observation_keep_time, expected_update_rate, max_observation_age, min_obstacle_height,
      max_obstacle_height;
    std::string topic, sensor_frame, data_type;
    bool inf_is_valid, clearing, marking;

    declareParameter(source + "." + "topic", rclcpp::ParameterValue(source));
    declareParameter(source + "." + "sensor_frame", rclcpp::ParameterValue(std::string("")));
    declareParameter(source + "." + "observation_persistence", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "expected_update_rate", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "max_observation_age", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "data_type", rclcpp::ParameterValue(std::string("LaserScan")));
    declareParameter(source + "." + "min_obstacle_height", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "max_obstacle_height", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "inf_is_valid", rclcpp::ParameterValue(false));
    declareParameter(source + "." + "marking", rclcpp::ParameterValue(true));
    declareParameter(source + "." + "clearing", rclcpp::ParameterValue(false));
    declareParameter(source + "." + "obstacle_max_range", rclcpp::ParameterValue(2.5));
    declareParameter(source + "." + "obstacle_min_range", rclcpp::ParameterValue(0.0));
    declareParameter(source + "." + "raytrace_max_range", rclcpp::ParameterValue(3.0));
    declareParameter(source + "." + "raytrace_min_range", rclcpp::ParameterValue(0.0));

    node->get_parameter(name_ + "." + source + "." + "topic", topic);
    node->get_parameter(name_ + "." + source + "." + "sensor_frame", sensor_frame);
    node->get_parameter(
      name_ + "." + source + "." + "observation_persistence",
      observation_keep_time);
    node->get_parameter(name_ + "." + source + "." + "expected_update_rate", expected_update_rate);
    node->get_parameter(name_ + "." + source + "." + "max_observation_age", max_observation_age);
    node->get_parameter(name_ + "." + source + "." + "data_type", data_type);
    node->get_parameter(name_ + "." + source + "." + "min_obstacle_height", min_obstacle_height);
    node->get_parameter(name_ + "." + source + "." + "max_obstacle_height", max_obstacle_height);
    node->get_parameter(name_ + "." + source + "." + "inf_is_valid", inf_is_valid);
    node->get_parameter(name_ + "." + source + "." + "marking", marking);
    node->get_parameter(name_ + "." + source + "." + "clearing", clearing);

    if (!(data_type == "PointCloud2" || data_type == "LaserScan")) {
      RCLCPP_FATAL(
        logger_,
        "Only topics that use point cloud2s or laser scans are "
        "currently supported");
      throw std::runtime_error(
              "Only topics that use point cloud2s or laser scans are currently "
              "supported");
    }

    // get the obstacle range for the sensor
    double obstacle_max_range, obstacle_min_range;
    node->get_parameter(name_ + "." + source + "." + "obstacle_max_range", obstacle_max_range);
    node->get_parameter(name_ + "." + source + "." + "obstacle_min_range", obstacle_min_range);

    // get the raytrace ranges for the sensor
    double raytrace_max_range, raytrace_min_range;
    node->get_parameter(name_ + "." + source + "." + "raytrace_min_range", raytrace_min_range);
    node->get_parameter(name_ + "." + source + "." + "raytrace_max_range", raytrace_max_range);

    RCLCPP_DEBUG(
      logger_, "Creating an observation buffer for source %s, topic %s, frame %s",
      source.c_str(), topic.c_str(), sensor_frame.c_str());

    // create an observation buffer
    observation_buffers_.push_back(
      std::shared_ptr<ObservationBufferWithBase>(
        new ObservationBufferWithBase(
          node, topic, observation_keep_time, expected_update_rate, min_obstacle_height,
          max_obstacle_height, obstacle_max_range, obstacle_min_range, raytrace_max_range,
          raytrace_min_range, *tf_, global_frame_, sensor_frame,
          tf2::durationFromSec(transform_tolerance), max_observation_age)));

    // check if we'll add this buffer to our marking observation buffers
    if (marking) {
      marking_buffers_.push_back(observation_buffers_.back());
    }

    // check if we'll also add this buffer to our clearing observation buffers
    if (clearing) {
      clearing_buffers_.push_back(observation_buffers_.back());
    }

    RCLCPP_DEBUG(
      logger_,
      "Created an observation buffer for source %s, topic %s, "
      "global frame: %s, "
      "expected update rate: %.2f, observation persistence: %.2f",
      source.c_str(), topic.c_str(), global_frame_.c_str(), expected_update_rate,
      observation_keep_time);

    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_sensor_data;
    custom_qos_profile.depth = 50;

    // create a callback for the topic
    if (data_type == "LaserScan") {
      auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan,
          rclcpp_lifecycle::LifecycleNode>>(
        node, topic, custom_qos_profile, sub_opt);
      sub->unsubscribe();

      auto filter = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>(
        *sub, *tf_, global_frame_, 50, node->get_node_logging_interface(),
        node->get_node_clock_interface(), tf2::durationFromSec(transform_tolerance));

      if (inf_is_valid) {
        filter->registerCallback(
          std::bind(
            &ObstacleDecayLayer::laserScanValidInfCallback, this,
            std::placeholders::_1, observation_buffers_.back()));

      } else {
        filter->registerCallback(
          std::bind(
            &ObstacleDecayLayer::laserScanCallback, this,
            std::placeholders::_1, observation_buffers_.back()));
      }

      observation_subscribers_.push_back(sub);

      observation_notifiers_.push_back(filter);
      observation_notifiers_.back()->setTolerance(rclcpp::Duration::from_seconds(0.05));

    } else {
      auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2,
          rclcpp_lifecycle::LifecycleNode>>(
        node, topic, custom_qos_profile, sub_opt);
      sub->unsubscribe();

      if (inf_is_valid) {
        RCLCPP_WARN(
          logger_,
          "obstacle_layer: inf_is_valid option is not applicable to "
          "PointCloud observations.");
      }

      auto filter = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
        *sub, *tf_, global_frame_, 50, node->get_node_logging_interface(),
        node->get_node_clock_interface(), tf2::durationFromSec(transform_tolerance));

      filter->registerCallback(
        std::bind(
          &ObstacleDecayLayer::pointCloud2Callback, this,
          std::placeholders::_1, observation_buffers_.back()));

      observation_subscribers_.push_back(sub);
      observation_notifiers_.push_back(filter);
    }

    if (sensor_frame != "") {
      std::vector<std::string> target_frames;
      target_frames.push_back(global_frame_);
      target_frames.push_back(sensor_frame);
      observation_notifiers_.back()->setTargetFrames(target_frames);
    }
  }
}

rcl_interfaces::msg::SetParametersResult ObstacleDecayLayer::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  rcl_interfaces::msg::SetParametersResult result;

  const std::string parameter_prefix = name_ + ".";
  const auto is_snapshot_startup_parameter =
    [this, &parameter_prefix](const std::string & parameter_name) {
      if (parameter_name == parameter_prefix + "snapshot_input_enabled") {
        // This switch is declared in both modes, but changing it cannot
        // create or destroy the subscription/evidence state safely.
        return true;
      }
      if (!snapshot_input_enabled_) {
        // The remaining snapshot parameters are not declared in legacy mode.
        return false;
      }
      return parameter_name == parameter_prefix + "snapshot_topic" ||
             parameter_name == parameter_prefix + "unreported_cells_are_free" ||
             parameter_name == parameter_prefix + "max_status_age" ||
             parameter_name == parameter_prefix + "max_capture_age" ||
             parameter_name == parameter_prefix + "future_tolerance" ||
             parameter_name == parameter_prefix + "snapshot_transform_tolerance" ||
             parameter_name == parameter_prefix + "max_grid_cells" ||
             parameter_name == parameter_prefix + "max_measurement_cells" ||
             parameter_name == parameter_prefix + "max_rasterized_cells" ||
             parameter_name == parameter_prefix + "hit_count_threshold" ||
             parameter_name == parameter_prefix + "miss_decrement" ||
             parameter_name == parameter_prefix + "clear_threshold" ||
             parameter_name == parameter_prefix + "max_persistent_cells";
    };

  // Validate the complete atomic update before changing any member. ROS rolls
  // back a rejected parameter transaction, so mutating a preceding parameter
  // here would otherwise leave the layer inconsistent with the node state.
  double proposed_min_obstacle_height = min_obstacle_height_;
  double proposed_max_obstacle_height = max_obstacle_height_;
  std::int64_t proposed_combination_method = combination_method_;
  for (const auto & parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();
    if (is_snapshot_startup_parameter(param_name)) {
      result.successful = false;
      result.reason = "snapshot obstacle parameters are startup-only";
      return result;
    }
    if (param_type == ParameterType::PARAMETER_BOOL) {
      if (snapshot_input_enabled_ && param_name == name_ + "." + "enabled") {
        result.successful = false;
        result.reason = "snapshot obstacle layer enable policy is startup-only";
        return result;
      }
      if (snapshot_input_enabled_ && param_name == name_ + "." + "footprint_clearing_enabled") {
        result.successful = false;
        result.reason = "snapshot footprint clearing policy is startup-only";
        return result;
      }
    }
    if (param_type == ParameterType::PARAMETER_DOUBLE) {
      if (param_name == parameter_prefix + "min_obstacle_height") {
        proposed_min_obstacle_height = parameter.as_double();
      } else if (param_name == parameter_prefix + "max_obstacle_height") {
        proposed_max_obstacle_height = parameter.as_double();
      }
    } else if (param_type == ParameterType::PARAMETER_INTEGER &&
      param_name == parameter_prefix + "combination_method")
    {
      proposed_combination_method = parameter.as_int();
    }
  }

  if (!std::isfinite(proposed_min_obstacle_height) ||
    !std::isfinite(proposed_max_obstacle_height) ||
    proposed_min_obstacle_height > proposed_max_obstacle_height)
  {
    result.successful = false;
    result.reason =
      "obstacle height limits must be finite and min_obstacle_height must not exceed "
      "max_obstacle_height";
    return result;
  }
  if (proposed_combination_method != 0 && proposed_combination_method != 1) {
    result.successful = false;
    result.reason = "combination_method must be 0 (overwrite) or 1 (maximum)";
    return result;
  }

  min_obstacle_height_ = proposed_min_obstacle_height;
  max_obstacle_height_ = proposed_max_obstacle_height;
  combination_method_ = static_cast<int>(proposed_combination_method);
  for (const auto & parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();

    if (param_type == ParameterType::PARAMETER_BOOL) {
      if (param_name == name_ + "." + "enabled") {
        enabled_ = parameter.as_bool();
      } else if (param_name == name_ + "." + "footprint_clearing_enabled") {
        footprint_clearing_enabled_ = parameter.as_bool();
      }
    }
  }

  result.successful = true;
  return result;
}

void ObstacleDecayLayer::laserScanCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr message,
  const std::shared_ptr<nav2_costmap_2d::ObservationBufferWithBase> & buffer)
{
  // project the laser into a point cloud
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = message->header;

  // project the scan into a point cloud
  try {
    projector_.transformLaserScanToPointCloud(message->header.frame_id, *message, cloud, *tf_);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(
      logger_,
      "High fidelity enabled, but TF returned a transform exception "
      "to frame %s: %s",
      global_frame_.c_str(), ex.what());
    projector_.projectLaser(*message, cloud);
  } catch (std::runtime_error & ex) {
    RCLCPP_WARN(
      logger_,
      "transformLaserScanToPointCloud error, it seems the message "
      "from laser is malformed."
      " Ignore this message. what(): %s",
      ex.what());
    buffer->invalidate();
    return;
  }

  // buffer the point cloud
  buffer->lock();
  buffer->bufferCloud(cloud);
  buffer->unlock();
}

void ObstacleDecayLayer::laserScanValidInfCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr raw_message,
  const std::shared_ptr<nav2_costmap_2d::ObservationBufferWithBase> & buffer)
{
  // Filter positive infinities ("Inf"s) to max_range.
  float epsilon = 0.0001;  // a tenth of a millimeter
  sensor_msgs::msg::LaserScan message = *raw_message;
  for (size_t i = 0; i < message.ranges.size(); i++) {
    float range = message.ranges[i];
    if (!std::isfinite(range) && range > 0) {
      message.ranges[i] = message.range_max - epsilon;
    }
  }

  // project the laser into a point cloud
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = message.header;

  // project the scan into a point cloud
  try {
    projector_.transformLaserScanToPointCloud(message.header.frame_id, message, cloud, *tf_);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(
      logger_,
      "High fidelity enabled, but TF returned a transform exception "
      "to frame %s: %s",
      global_frame_.c_str(), ex.what());
    projector_.projectLaser(message, cloud);
  } catch (std::runtime_error & ex) {
    RCLCPP_WARN(
      logger_,
      "transformLaserScanToPointCloud error, it seems the message "
      "from laser is malformed."
      " Ignore this message. what(): %s",
      ex.what());
    buffer->invalidate();
    return;
  }

  // buffer the point cloud
  buffer->lock();
  buffer->bufferCloud(cloud);
  buffer->unlock();
}

void ObstacleDecayLayer::pointCloud2Callback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr message,
  const std::shared_ptr<ObservationBufferWithBase> & buffer)
{
  // buffer the point cloud
  buffer->lock();
  buffer->bufferCloud(*message);
  buffer->unlock();
}

void ObstacleDecayLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (rolling_window_) {
    double new_origin_x = 0.0;
    double new_origin_y = 0.0;
    layered_costmap_->getRollingWindowOrigin(
      robot_x, robot_y, robot_yaw, new_origin_x, new_origin_y);
    updateOrigin(new_origin_x, new_origin_y);
  }
  if (!enabled_) {
    return;
  }
  if (snapshot_input_enabled_) {
    useExtraBounds(min_x, min_y, max_x, max_y);
    updateSnapshotBounds(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);
    return;
  }
  useExtraBounds(min_x, min_y, max_x, max_y);

  current_ = false;
  update_pending_ = false;
  pending_observations_.clear();
  std::vector<Observation> observations, clearing_observations;
  const bool marking_current = getMarkingObservations(observations, &pending_observations_);
  const bool clearing_current =
    getClearingObservations(clearing_observations, &pending_observations_);

  // Always merge the retained grid, including while observations are unavailable.
  *min_x = robot_x - getSizeInMetersX();
  *min_y = robot_y - getSizeInMetersY();
  *max_x = robot_x + getSizeInMetersX();
  *max_y = robot_y + getSizeInMetersY();
  if (!marking_current || !clearing_current || observations.empty()) {
    return;
  }
  tmp_costmap_.resize(size_x_ * size_y_);

  // A successful snapshot replaces the previous grid. An empty *cloud* is a
  // valid observation; an empty observation list is not evidence of free space.
  // Rebuild from the retained latest snapshot (observation_persistence == 0).
  for (unsigned int x = 0; x < size_x_; x++) {
    for (unsigned int y = 0; y < size_y_; y++) {
      unsigned int index = getIndex(x, y);
      tmp_costmap_.at(index) = Grid();
      costmap_[index] = nav2_costmap_2d::FREE_SPACE;
    }
  }

  // place the new obstacles into a priority queue... each with a priority of
  // zero to begin with
  for (std::vector<Observation>::const_iterator it = observations.begin(); it != observations.end();
    ++it)
  {
    const Observation & obs = *it;

    const sensor_msgs::msg::PointCloud2 & cloud = *(obs.cloud_);

    double sq_obstacle_max_range = obs.obstacle_max_range_ * obs.obstacle_max_range_;
    double sq_obstacle_min_range = obs.obstacle_min_range_ * obs.obstacle_min_range_;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      double px = *iter_x, py = *iter_y, pz = *iter_z;

      // if the obstacle is too low, we won't add it
      if ((pz - obs.origin_.z) < min_obstacle_height_) {
        RCLCPP_DEBUG(logger_, "The point is too low");
        continue;
      }

      // if the obstacle is too high or too far away from the robot we won't add
      // it
      if ((pz - obs.origin_.z) > max_obstacle_height_) {
        RCLCPP_DEBUG(logger_, "The point is too high");
        continue;
      }

      // compute the squared distance from the hitpoint to the pointcloud's
      // origin
      double sq_dist = (px - obs.origin_.x) * (px - obs.origin_.x) +
        (py - obs.origin_.y) * (py - obs.origin_.y) +
        (pz - obs.origin_.z) * (pz - obs.origin_.z);

      // if the point is far enough away... we won't consider it
      if (sq_dist >= sq_obstacle_max_range) {
        RCLCPP_DEBUG(logger_, "The point is too far away");
        continue;
      }

      // if the point is too close, do not conisder it
      if (sq_dist < sq_obstacle_min_range) {
        RCLCPP_DEBUG(logger_, "The point is too close");
        continue;
      }

      // now we need to compute the map coordinates for the observation
      unsigned int mx, my;
      if (!worldToMap(px, py, mx, my)) {
        RCLCPP_DEBUG(logger_, "Computing map coords failed");
        continue;
      }

      // unsigned int index = getIndex(mx, my);
      // costmap_[index] = LETHAL_OBSTACLE;
      // touch(px, py, min_x, min_y, max_x, max_y);

      unsigned int index = getIndex(mx, my);
      Grid & grid = tmp_costmap_.at(index);
      ++grid.count_;
      touch(px, py, &grid.min_x_, &grid.min_y_, &grid.max_x_, &grid.max_y_);
    }
  }

  // X. Copy temporary buffer.
  for (size_t idx_x = 0; idx_x < size_x_; ++idx_x) {
    for (size_t idx_y = 0; idx_y < size_y_; ++idx_y) {
      unsigned int index = getIndex(idx_x, idx_y);
      const Grid & grid = tmp_costmap_.at(index);
      if (obstacle_pointcount_ <= grid.count_) {
        costmap_[index] = LETHAL_OBSTACLE;
        touch(grid.min_x_, grid.min_y_, min_x, min_y, max_x, max_y);
        touch(grid.max_x_, grid.max_y_, min_x, min_y, max_x, max_y);
      }
    }
  }

  updateFootprint(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);
  update_pending_ = true;
}

void ObstacleDecayLayer::updateFootprint(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x,
  double * max_y)
{
  if (!footprint_clearing_enabled_) {
    return;
  }
  transformFootprint(robot_x, robot_y, robot_yaw, getFootprint(), transformed_footprint_);

  for (unsigned int i = 0; i < transformed_footprint_.size(); i++) {
    touch(transformed_footprint_[i].x, transformed_footprint_[i].y, min_x, min_y, max_x, max_y);
  }
}

void ObstacleDecayLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j,
  int max_i, int max_j)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!enabled_) {
    return;
  }

  if (update_pending_ && footprint_clearing_enabled_ && !snapshot_input_enabled_) {
    setConvexPolygonCost(transformed_footprint_, nav2_costmap_2d::FREE_SPACE);
  }

  switch (combination_method_) {
    case 0:  // Overwrite
      updateWithOverwrite(master_grid, min_i, min_j, max_i, max_j);
      break;
    case 1:  // Maximum
      updateWithMax(master_grid, min_i, min_j, max_i, max_j);
      break;
    default:  // Nothing
      current_ = false;
      return;
  }
  if (update_pending_) {
    if (!snapshot_input_enabled_) {
      applied_observations_ = pending_observations_;
    }
    update_pending_ = false;
    current_ = true;
  }
}

bool ObstacleDecayLayer::isCurrent() const
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (snapshot_input_enabled_) {
    return snapshotCurrentLocked(clock_->now().nanoseconds());
  }
  if (!current_) {
    return false;
  }
  for (const auto & snapshot : applied_observations_) {
    if (!snapshot.buffer->isCurrent(snapshot.observation)) {
      return false;
    }
  }
  return true;
}

void ObstacleDecayLayer::addStaticObservation(
  nav2_costmap_2d::Observation & obs, bool marking,
  bool clearing)
{
  if (marking) {
    static_marking_observations_.push_back(obs);
  }
  if (clearing) {
    static_clearing_observations_.push_back(obs);
  }
}

void ObstacleDecayLayer::clearStaticObservations(bool marking, bool clearing)
{
  if (marking) {
    static_marking_observations_.clear();
  }
  if (clearing) {
    static_clearing_observations_.clear();
  }
}

bool ObstacleDecayLayer::getMarkingObservations(
  std::vector<Observation> & marking_observations,
  std::vector<BufferedFreshness> * snapshots) const
{
  bool current = true;
  // get the marking observations
  for (unsigned int i = 0; i < marking_buffers_.size(); ++i) {
    marking_buffers_[i]->lock();
    ObservationBufferWithBase::Freshness observation;
    marking_buffers_[i]->getObservations(marking_observations, &observation);
    if (snapshots) {
      snapshots->push_back({marking_buffers_[i], observation});
    }
    current = marking_buffers_[i]->isCurrent() && current;
    marking_buffers_[i]->unlock();
  }
  marking_observations.insert(
    marking_observations.end(), static_marking_observations_.begin(),
    static_marking_observations_.end());
  return current;
}

bool ObstacleDecayLayer::getClearingObservations(
  std::vector<Observation> & clearing_observations,
  std::vector<BufferedFreshness> * snapshots) const
{
  bool current = true;
  // get the clearing observations
  for (unsigned int i = 0; i < clearing_buffers_.size(); ++i) {
    clearing_buffers_[i]->lock();
    ObservationBufferWithBase::Freshness observation;
    clearing_buffers_[i]->getObservations(clearing_observations, &observation);
    if (snapshots) {
      snapshots->push_back({clearing_buffers_[i], observation});
    }
    current = clearing_buffers_[i]->isCurrent() && current;
    clearing_buffers_[i]->unlock();
  }
  clearing_observations.insert(
    clearing_observations.end(), static_clearing_observations_.begin(),
    static_clearing_observations_.end());
  return current;
}

void ObstacleDecayLayer::raytraceFreespace(
  const Observation & clearing_observation, double * min_x,
  double * min_y, double * max_x, double * max_y)
{
  double ox = clearing_observation.origin_.x;
  double oy = clearing_observation.origin_.y;
  const sensor_msgs::msg::PointCloud2 & cloud = *(clearing_observation.cloud_);

  // get the map coordinates of the origin of the sensor
  unsigned int x0, y0;
  if (!worldToMap(ox, oy, x0, y0)) {
    RCLCPP_WARN(
      logger_,
      "Sensor origin at (%.2f, %.2f) is out of map bounds (%.2f, "
      "%.2f) to (%.2f, %.2f). "
      "The costmap cannot raytrace for it.",
      ox, oy, origin_x_, origin_y_, origin_x_ + getSizeInMetersX(),
      origin_y_ + getSizeInMetersY());
    return;
  }

  // we can pre-compute the enpoints of the map outside of the inner loop...
  // we'll need these later
  double origin_x = origin_x_, origin_y = origin_y_;
  double map_end_x = origin_x + size_x_ * resolution_;
  double map_end_y = origin_y + size_y_ * resolution_;

  touch(ox, oy, min_x, min_y, max_x, max_y);

  // for each point in the cloud, we want to trace a line from the origin
  // and clear obstacles along it
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
    double wx = *iter_x;
    double wy = *iter_y;

    // now we also need to make sure that the enpoint we're raytracing
    // to isn't off the costmap and scale if necessary
    double a = wx - ox;
    double b = wy - oy;

    // the minimum value to raytrace from is the origin
    if (wx < origin_x) {
      double t = (origin_x - ox) / a;
      wx = origin_x;
      wy = oy + b * t;
    }
    if (wy < origin_y) {
      double t = (origin_y - oy) / b;
      wx = ox + a * t;
      wy = origin_y;
    }

    // the maximum value to raytrace to is the end of the map
    if (wx > map_end_x) {
      double t = (map_end_x - ox) / a;
      wx = map_end_x - .001;
      wy = oy + b * t;
    }
    if (wy > map_end_y) {
      double t = (map_end_y - oy) / b;
      wx = ox + a * t;
      wy = map_end_y - .001;
    }

    // now that the vector is scaled correctly... we'll get the map coordinates
    // of its endpoint
    unsigned int x1, y1;

    // check for legality just in case
    if (!worldToMap(wx, wy, x1, y1)) {
      continue;
    }

    unsigned int cell_raytrace_max_range = cellDistance(clearing_observation.raytrace_max_range_);
    unsigned int cell_raytrace_min_range = cellDistance(clearing_observation.raytrace_min_range_);
    MarkCell marker(costmap_, FREE_SPACE);
    // and finally... we can execute our trace to clear obstacles along that
    // line
    raytraceLine(marker, x0, y0, x1, y1, cell_raytrace_max_range, cell_raytrace_min_range);

    updateRaytraceBounds(
      ox, oy, wx, wy, clearing_observation.raytrace_max_range_,
      clearing_observation.raytrace_min_range_, min_x, min_y, max_x, max_y);
  }
}

void ObstacleDecayLayer::activate()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  current_ = false;
  applied_observations_.clear();
  pending_observations_.clear();
  update_pending_ = false;
  if (snapshot_input_enabled_) {
    snapshot_active_ = true;
    latest_snapshot_.reset();
    return;
  }
  for (auto & notifier : observation_notifiers_) {
    notifier->clear();
  }

  // X. Temporary buffer
  tmp_costmap_.resize(size_x_ * size_y_, Grid());

  // if we're stopped we need to re-subscribe to topics
  for (unsigned int i = 0; i < observation_subscribers_.size(); ++i) {
    if (observation_subscribers_[i] != NULL) {
      observation_subscribers_[i]->subscribe();
    }
  }
  resetBuffersLastUpdated();
}

void ObstacleDecayLayer::deactivate()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  current_ = false;
  if (snapshot_input_enabled_) {
    snapshot_active_ = false;
    latest_snapshot_.reset();
    update_pending_ = false;
    return;
  }
  for (unsigned int i = 0; i < observation_subscribers_.size(); ++i) {
    if (observation_subscribers_[i] != NULL) {
      observation_subscribers_[i]->unsubscribe();
    }
  }

  // X. Clear temporary buffer.
  tmp_costmap_.clear();
}

void ObstacleDecayLayer::updateRaytraceBounds(
  double ox, double oy, double wx, double wy,
  double max_range, double min_range, double * min_x,
  double * min_y, double * max_x, double * max_y)
{
  double dx = wx - ox, dy = wy - oy;
  double full_distance = hypot(dx, dy);
  if (full_distance < min_range) {
    return;
  }
  double scale = std::min(1.0, max_range / full_distance);
  double ex = ox + dx * scale, ey = oy + dy * scale;
  touch(ex, ey, min_x, min_y, max_x, max_y);
}

void ObstacleDecayLayer::reset()
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  resetMaps();
  current_ = false;
  update_pending_ = false;
  applied_observations_.clear();
  pending_observations_.clear();
  if (snapshot_input_enabled_) {
    latest_snapshot_.reset();
    has_snapshot_watermark_ = false;
    watermark_generation_ = 0U;
    watermark_sequence_ = 0U;
    last_evidence_generation_ = 0U;
    last_valid_capture_ns_ = 0;
    last_valid_status_ns_ = 0;
    temporal_evidence_->reset();
    const unsigned char background = unreported_cells_are_free_ ? FREE_SPACE : NO_INFORMATION;
    std::fill(costmap_, costmap_ + size_x_ * size_y_, background);
    return;
  }
  resetBuffersLastUpdated();
}

void ObstacleDecayLayer::resetBuffersLastUpdated()
{
  for (unsigned int i = 0; i < observation_buffers_.size(); ++i) {
    if (observation_buffers_[i]) {
      observation_buffers_[i]->resetLastUpdated();
    }
  }
}

}  // namespace nav2_costmap_2d
