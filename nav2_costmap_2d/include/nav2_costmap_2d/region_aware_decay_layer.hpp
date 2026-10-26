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
#ifndef NAV2_COSTMAP_2D__REGION_AWARE_DECAY_LAYER_HPP_
#define NAV2_COSTMAP_2D__REGION_AWARE_DECAY_LAYER_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#include "tf2_ros/message_filter.h"
#pragma GCC diagnostic pop
#include "message_filters/subscriber.h"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/observation_buffer_with_base.hpp"
#include "nav2_costmap_2d/footprint.hpp"

namespace nav2_costmap_2d
{

/**
 * A separate, opt-in replacement for ObstacleDecayLayer. Each accepted pointcloud
 * advances cell histories once when the costmap timer drains the queue, before
 * publishing the final result.
 * The current implementation accepts one PointCloud2 marking source, with no
 * raytracing and observation_persistence == 0. Unsupported modes fail startup.
 */
class RegionAwareDecayLayer : public CostmapLayer
{
public:
  RegionAwareDecayLayer() = default;
  ~RegionAwareDecayLayer() override;

  void onInitialize() override;
  void matchSize() override;
  void updateOrigin(double new_origin_x, double new_origin_y) override;
  void clearArea(int start_x, int start_y, int end_x, int end_y, bool invert) override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j) override;
  void activate() override;
  void deactivate() override;
  void reset() override;
  bool isCurrent() const override;
  bool isClearable() override {return true;}

  void pointCloud2Callback(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message,
    const std::shared_ptr<ObservationBufferWithBase> & buffer);
  void resetBuffersLastUpdated();
  rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    std::vector<rclcpp::Parameter> parameters);

protected:
  struct BufferedFreshness
  {
    std::shared_ptr<ObservationBufferWithBase> buffer;
    ObservationBufferWithBase::Freshness observation;
  };

  // Only real samples are stored. count is hits in a and misses in b.
  struct History
  {
    std::vector<uint8_t> values;
    size_t next = 0;
    unsigned int count = 0;
    void append(bool evidence, unsigned int window);
    void clear();
  };

  struct CellHistory
  {
    bool occupied = false;
    bool latest_hit = false;
    int64_t last_seen_ns = 0;
    History a;
    History b;
  };

  struct QueuedObservation
  {
    // Observation's legacy assignment operator leaks its previous cloud, so
    // retain immutable copied observations through shared ownership instead.
    std::shared_ptr<const Observation> observation;
    BufferedFreshness freshness;
    std::string source;
  };

  struct BlindRegion
  {
    std::string name;
    double x_min, x_max, y_min, y_max, hold_seconds;
  };

  std::vector<std::shared_ptr<ObservationBufferWithBase>> observation_buffers_;
  std::vector<CellHistory> cells_;
  std::deque<std::shared_ptr<const QueuedObservation>> queue_;
  uint64_t queue_overflows_ = 0;
  uint64_t stale_dropped_ = 0;
  uint64_t observations_processed_ = 0;

private:
  void clearInput();
  void filterClearedObservations(int start_x, int start_y, int end_x, int end_y, bool invert);
  std::shared_ptr<const QueuedObservation> filterClearedObservation(
    const std::shared_ptr<const QueuedObservation> & event,
    int start_x, int start_y, int end_x, int end_y, bool invert) const;
  void removeClearedPoints(
    sensor_msgs::msg::PointCloud2 & cloud,
    int start_x, int start_y, int end_x, int end_y, bool invert) const;
  void clearHistories(bool retain_occupied);
  bool checkClock();
  void createHitMask(const Observation & observation);
  void applyObservation(
    const QueuedObservation & event, double robot_x, double robot_y,
    double robot_yaw, int64_t observation_ns);
  bool mayFree(
    size_t index, double robot_x, double robot_y, double robot_yaw, int64_t observation_ns) const;
  double holdingTime(
    size_t index, double robot_x, double robot_y, double robot_yaw,
    std::string * region_name = nullptr) const;
  void rebuildGrid();
  void updateFootprint(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y);
  void clearFootprintHistory();
  void logDiagnostics(size_t processed, double elapsed_ms);

  std::vector<std::shared_ptr<message_filters::SubscriberBase<rclcpp_lifecycle::LifecycleNode>>>
  observation_subscribers_;
  std::vector<std::shared_ptr<tf2_ros::MessageFilterBase>> observation_notifiers_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<const QueuedObservation> latest_snapshot_;
  std::vector<BufferedFreshness> pending_observations_;
  std::vector<BufferedFreshness> applied_observations_;
  std::vector<geometry_msgs::msg::Point> transformed_footprint_;
  std::vector<BlindRegion> blind_regions_;
  std::vector<unsigned int> point_counts_;
  std::string source_name_;
  std::string global_frame_;
  unsigned int occupy_min_hits_ = 1;
  unsigned int occupy_window_ = 1;
  unsigned int free_min_misses_ = 1;
  unsigned int free_window_ = 1;
  size_t observation_queue_size_ = 64;
  int obstacle_pointcount_ = 1;
  int combination_method_ = 1;
  double min_obstacle_height_ = 0.0;
  double max_obstacle_height_ = 2.0;
  bool footprint_clearing_enabled_ = true;
  bool rolling_window_ = false;
  bool update_pending_ = false;
  bool active_ = false;
  int64_t last_clock_ns_ = -1;
  int64_t last_enqueued_capture_ns_ = 0;
  uint64_t last_enqueued_generation_ = 0;
  size_t queue_high_watermark_ = 0;
  size_t pending_processed_ = 0;
  double bounds_update_ms_ = 0.0;
  double maximum_update_ms_ = 0.0;
  uint64_t clock_resets_ = 0;
};

}  // namespace nav2_costmap_2d
#endif  // NAV2_COSTMAP_2D__REGION_AWARE_DECAY_LAYER_HPP_
