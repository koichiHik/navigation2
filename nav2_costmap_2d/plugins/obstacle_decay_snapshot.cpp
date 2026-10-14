// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "builtin_interfaces/msg/time.hpp"
#include "camera_localization_msgs/msg/obstacle_measurement_cell.hpp"
#include "camera_localization_msgs/msg/obstacle_status.hpp"
#include "nav2_costmap_2d/obstacle_decay_layer.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace nav2_costmap_2d
{
namespace
{

using Snapshot = camera_localization_msgs::msg::ObstacleGridSnapshot;
using Cell = camera_localization_msgs::msg::ObstacleMeasurementCell;
using Status = camera_localization_msgs::msg::ObstacleStatus;

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

std::optional<std::int64_t> TimeToNanoseconds(const builtin_interfaces::msg::Time & stamp) noexcept
{
  if (stamp.sec < 0 || stamp.nanosec >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    return std::nullopt;
  }
  const std::int64_t seconds = static_cast<std::int64_t>(stamp.sec);
  const std::int64_t nanoseconds = static_cast<std::int64_t>(stamp.nanosec);
  if (seconds > (std::numeric_limits<std::int64_t>::max() - nanoseconds) / kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return seconds * kNanosecondsPerSecond + nanoseconds;
}

bool LaterThanTolerance(
  const std::int64_t value, const std::int64_t now,
  const std::int64_t tolerance) noexcept
{
  return value > now && value - now > tolerance;
}

std::int64_t SecondsToNanoseconds(const double seconds, const bool allow_zero = false)
{
  if (!std::isfinite(seconds) || seconds < 0.0 || (!allow_zero && seconds == 0.0) ||
    seconds > static_cast<double>(std::numeric_limits<std::int64_t>::max()) / 1.0e9)
  {
    throw std::invalid_argument("invalid obstacle snapshot duration");
  }
  return static_cast<std::int64_t>(std::llround(seconds * 1.0e9));
}

std::array<EvidencePoint2d, 4> TransformSourceCell(
  const Snapshot & snapshot, const std::int32_t ix,
  const std::int32_t iy,
  const tf2::Transform & transform) noexcept
{
  const double min_x = snapshot.grid_origin_x + static_cast<double>(ix) * snapshot.grid_resolution;
  const double min_y = snapshot.grid_origin_y + static_cast<double>(iy) * snapshot.grid_resolution;
  const double max_x = min_x + snapshot.grid_resolution;
  const double max_y = min_y + snapshot.grid_resolution;
  const std::array<tf2::Vector3, 4> source{
    {{min_x, min_y, 0.0}, {max_x, min_y, 0.0}, {max_x, max_y, 0.0}, {min_x, max_y, 0.0}}};
  std::array<EvidencePoint2d, 4> result{};
  for (std::size_t index = 0U; index < source.size(); ++index) {
    const tf2::Vector3 transformed = transform * source[index];
    result[index] = {transformed.x(), transformed.y()};
  }
  return result;
}

}  // namespace

bool ObstacleDecayLayer::initializeSnapshotInput(const std::string & legacy_topics)
{
  declareParameter("snapshot_input_enabled", rclcpp::ParameterValue(false));
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }
  node->get_parameter(name_ + ".snapshot_input_enabled", snapshot_input_enabled_);
  if (!snapshot_input_enabled_) {
    return false;
  }

  declareParameter(
    "snapshot_topic",
    rclcpp::ParameterValue(std::string("/stereo_obstacles/grid_snapshot")));
  declareParameter("unreported_cells_are_free", rclcpp::ParameterValue(true));
  declareParameter("max_status_age", rclcpp::ParameterValue(0.5));
  declareParameter("max_capture_age", rclcpp::ParameterValue(0.5));
  declareParameter("future_tolerance", rclcpp::ParameterValue(0.05));
  declareParameter("snapshot_transform_tolerance", rclcpp::ParameterValue(0.2));
  declareParameter("max_grid_cells", rclcpp::ParameterValue(std::int64_t{1'000'000}));
  declareParameter("max_measurement_cells", rclcpp::ParameterValue(std::int64_t{100'000}));
  declareParameter("max_rasterized_cells", rclcpp::ParameterValue(std::int64_t{500'000}));
  declareParameter("hit_count_threshold", rclcpp::ParameterValue(std::int64_t{2}));
  declareParameter("miss_decrement", rclcpp::ParameterValue(std::int64_t{1}));
  declareParameter("clear_threshold", rclcpp::ParameterValue(std::int64_t{0}));
  declareParameter("max_persistent_cells", rclcpp::ParameterValue(std::int64_t{500'000}));

  double maximum_status_age_sec = 0.0;
  double maximum_capture_age_sec = 0.0;
  double future_tolerance_sec = 0.0;
  std::int64_t maximum_grid_cells = 0;
  std::int64_t maximum_measurement_cells = 0;
  std::int64_t maximum_rasterized_cells = 0;
  std::int64_t hit_count_threshold = 0;
  std::int64_t miss_decrement = 0;
  std::int64_t clear_threshold = 0;
  std::int64_t maximum_persistent_cells = 0;
  node->get_parameter(name_ + ".snapshot_topic", snapshot_topic_);
  node->get_parameter(name_ + ".unreported_cells_are_free", unreported_cells_are_free_);
  node->get_parameter(name_ + ".max_status_age", maximum_status_age_sec);
  node->get_parameter(name_ + ".max_capture_age", maximum_capture_age_sec);
  node->get_parameter(name_ + ".future_tolerance", future_tolerance_sec);
  node->get_parameter(name_ + ".snapshot_transform_tolerance", snapshot_transform_tolerance_sec_);
  node->get_parameter(name_ + ".max_grid_cells", maximum_grid_cells);
  node->get_parameter(name_ + ".max_measurement_cells", maximum_measurement_cells);
  node->get_parameter(name_ + ".max_rasterized_cells", maximum_rasterized_cells);
  node->get_parameter(name_ + ".hit_count_threshold", hit_count_threshold);
  node->get_parameter(name_ + ".miss_decrement", miss_decrement);
  node->get_parameter(name_ + ".clear_threshold", clear_threshold);
  node->get_parameter(name_ + ".max_persistent_cells", maximum_persistent_cells);

  const auto positive_size = [](const std::int64_t value, const char * name) {
      if (value <= 0 ||
        static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max())
      {
        throw std::invalid_argument(std::string("invalid ") + name);
      }
      return static_cast<std::size_t>(value);
    };
  const auto bounded_int = [](const std::int64_t value, const char * name) {
      if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("invalid ") + name);
      }
      return static_cast<int>(value);
    };

  maximum_status_age_ns_ = SecondsToNanoseconds(maximum_status_age_sec);
  maximum_capture_age_ns_ = SecondsToNanoseconds(maximum_capture_age_sec);
  future_tolerance_ns_ = SecondsToNanoseconds(future_tolerance_sec, true);
  maximum_grid_cells_ = positive_size(maximum_grid_cells, "max_grid_cells");
  maximum_measurement_cells_ = positive_size(maximum_measurement_cells, "max_measurement_cells");
  maximum_rasterized_cells_ = positive_size(maximum_rasterized_cells, "max_rasterized_cells");
  temporal_evidence_parameters_.hit_count_threshold =
    positive_size(hit_count_threshold, "hit_count_threshold");
  temporal_evidence_parameters_.miss_decrement = bounded_int(miss_decrement, "miss_decrement");
  temporal_evidence_parameters_.clear_threshold = bounded_int(clear_threshold, "clear_threshold");
  temporal_evidence_parameters_.maximum_persistent_cells =
    positive_size(maximum_persistent_cells, "max_persistent_cells");
  if (snapshot_topic_.empty() || !std::isfinite(snapshot_transform_tolerance_sec_) ||
    snapshot_transform_tolerance_sec_ < 0.0 || !temporal_evidence_parameters_.valid())
  {
    throw std::invalid_argument("invalid obstacle snapshot configuration");
  }
  if (!footprint_clearing_enabled_) {
    throw std::invalid_argument("snapshot obstacle input requires footprint clearing");
  }
  if (!legacy_topics.empty()) {
    RCLCPP_WARN(
      logger_,
      "Snapshot obstacle input is enabled; legacy observation_sources '%s' are ignored",
      legacy_topics.c_str());
  }

  evidence_resolution_m_ = getResolution();
  if (!std::isfinite(evidence_resolution_m_) || evidence_resolution_m_ <= 0.0) {
    throw std::invalid_argument("invalid Costmap resolution for temporal obstacle evidence");
  }
  temporal_evidence_ = std::make_unique<TemporalObstacleEvidence>(temporal_evidence_parameters_);
  auto options = rclcpp::SubscriptionOptions();
  options.callback_group = callback_group_;
  snapshot_subscription_ = node->create_subscription<Snapshot>(
    snapshot_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
    std::bind(&ObstacleDecayLayer::obstacleGridSnapshotCallback, this, std::placeholders::_1),
    options);
  RCLCPP_INFO(
    logger_,
    "Accepted obstacle snapshot input enabled on %s: %zu distinct hits to occupy, miss decrement "
    "%d, clear threshold %d",
    snapshot_topic_.c_str(), temporal_evidence_parameters_.hit_count_threshold,
    temporal_evidence_parameters_.miss_decrement, temporal_evidence_parameters_.clear_threshold);
  return true;
}

bool ObstacleDecayLayer::validateSnapshotLocked(
  const Snapshot & snapshot,
  const std::int64_t now_ns) const noexcept
{
  try {
    if (snapshot.schema_version != Snapshot::SCHEMA_VERSION ||
      snapshot.free_space_semantics != Snapshot::FREE_SPACE_HITS_ONLY ||
      !snapshot.coverage.empty() || !snapshot.measurement_complete ||
      snapshot.header.frame_id.empty())
    {
      return false;
    }

    const Status & status = snapshot.status;
    const auto header_stamp = TimeToNanoseconds(snapshot.header.stamp);
    const auto receive_stamp = TimeToNanoseconds(snapshot.receive_stamp);
    const auto status_stamp = TimeToNanoseconds(status.status_stamp);
    const auto capture_stamp = TimeToNanoseconds(status.capture_stamp);
    if (status.schema_version != Status::SCHEMA_VERSION || status.run_generation == 0U ||
      status.status_timestamp_ns <= 0 || status.capture_timestamp_ns <= 0 ||
      status.status_timestamp_ns < status.capture_timestamp_ns ||
      header_stamp != status.capture_timestamp_ns ||
      capture_stamp != status.capture_timestamp_ns ||
      receive_stamp != snapshot.receive_timestamp_ns ||
      status_stamp != status.status_timestamp_ns ||
      snapshot.receive_timestamp_ns != status.status_timestamp_ns ||
      status.processing_state != Status::PROCESSING_VALID || !status.process_alive ||
      !status.image_fresh || !status.physical_stereo_complete || !status.obstacle_complete ||
      status.queue_faulted || !status.calibration_match || !status.sgm_profile_match ||
      !status.authority_valid || status.required_camera_mask == 0U ||
      (status.present_camera_mask & status.required_camera_mask) != status.required_camera_mask ||
      status.required_pair_count == 0U ||
      status.processed_pair_count != status.required_pair_count ||
      status.obstacle_frame_id != snapshot.header.frame_id ||
      LaterThanTolerance(status.status_timestamp_ns, now_ns, future_tolerance_ns_) ||
      LaterThanTolerance(status.capture_timestamp_ns, now_ns, future_tolerance_ns_) ||
      now_ns <= 0 || now_ns - status.status_timestamp_ns > maximum_status_age_ns_ ||
      now_ns - status.capture_timestamp_ns > maximum_capture_age_ns_)
    {
      return false;
    }

    const std::uint64_t grid_cell_count =
      static_cast<std::uint64_t>(snapshot.grid_width) * snapshot.grid_height;
    const double grid_max_x = snapshot.grid_origin_x +
      static_cast<double>(snapshot.grid_width) * snapshot.grid_resolution;
    const double grid_max_y = snapshot.grid_origin_y +
      static_cast<double>(snapshot.grid_height) * snapshot.grid_resolution;
    if (!std::isfinite(snapshot.grid_resolution) || snapshot.grid_resolution <= 0.0F ||
      !std::isfinite(snapshot.grid_origin_x) || !std::isfinite(snapshot.grid_origin_y) ||
      !std::isfinite(grid_max_x) || !std::isfinite(grid_max_y) || snapshot.grid_width == 0U ||
      snapshot.grid_height == 0U ||
      snapshot.grid_width >
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      snapshot.grid_height >
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      grid_cell_count > maximum_grid_cells_ ||
      snapshot.cluster_accepted_cells.size() > maximum_measurement_cells_ ||
      snapshot.cluster_empty != snapshot.cluster_accepted_cells.empty() ||
      status.point_count != snapshot.cluster_accepted_cells.size() ||
      status.occupied_cell_count != snapshot.cluster_accepted_cells.size())
    {
      return false;
    }

    std::set<std::pair<std::int32_t, std::int32_t>> unique_cells;
    for (const Cell & cell : snapshot.cluster_accepted_cells) {
      if (cell.ix < 0 || cell.iy < 0 ||
        static_cast<std::uint32_t>(cell.ix) >= snapshot.grid_width ||
        static_cast<std::uint32_t>(cell.iy) >= snapshot.grid_height ||
        cell.stage != Cell::STAGE_CLUSTER_ACCEPTED || !std::isfinite(cell.z_min) ||
        !std::isfinite(cell.z_max) || cell.z_min > cell.z_max || cell.point_count == 0U ||
        !std::isfinite(cell.quality) || cell.quality < 0.0F || cell.quality > 1.0F ||
        cell.contributing_source_mask == 0U || !unique_cells.emplace(cell.ix, cell.iy).second)
      {
        return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

void ObstacleDecayLayer::invalidateSnapshotLocked() noexcept
{
  latest_snapshot_.reset();
  update_pending_ = false;
  current_ = false;
}

void ObstacleDecayLayer::obstacleGridSnapshotCallback(Snapshot::ConstSharedPtr message)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  if (!snapshot_input_enabled_ || !snapshot_active_) {
    return;
  }
  if (!message) {
    invalidateSnapshotLocked();
    return;
  }

  // STOP/invalid snapshots may deliberately reuse the last valid sequence.
  // They must revoke currentness immediately, but must never poison the valid
  // ordering watermark or mutate retained evidence.
  if (!validateSnapshotLocked(*message, clock_->now().nanoseconds())) {
    invalidateSnapshotLocked();
    return;
  }

  const std::uint64_t generation = message->status.run_generation;
  const std::uint64_t sequence = message->status.frame_sequence;
  if (has_snapshot_watermark_) {
    if (generation < watermark_generation_ ||
      (generation == watermark_generation_ && sequence <= watermark_sequence_))
    {
      // Old deliveries and exact retries are harmless and cannot add a hit.
      return;
    }
  }
  has_snapshot_watermark_ = true;
  watermark_generation_ = generation;
  watermark_sequence_ = sequence;
  latest_snapshot_ = std::move(message);
}

void ObstacleDecayLayer::updateSnapshotBounds(
  const double robot_x, const double robot_y,
  const double robot_yaw, double * min_x, double * min_y,
  double * max_x, double * max_y)
{
  *min_x = std::min(*min_x, robot_x - getSizeInMetersX());
  *min_y = std::min(*min_y, robot_y - getSizeInMetersY());
  *max_x = std::max(*max_x, robot_x + getSizeInMetersX());
  *max_y = std::max(*max_y, robot_y + getSizeInMetersY());
  updateFootprint(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);

  const auto render_with_footprint_mask = [this]() {
      renderTemporalEvidenceLocked();
      if (footprint_clearing_enabled_) {
        setConvexPolygonCost(transformed_footprint_, FREE_SPACE);
      }
    };

  if (!latest_snapshot_) {
    render_with_footprint_mask();
    return;
  }
  const Snapshot & snapshot = *latest_snapshot_;
  const std::int64_t now_ns = clock_->now().nanoseconds();
  if (!validateSnapshotLocked(snapshot, now_ns) || getResolution() != evidence_resolution_m_) {
    invalidateSnapshotLocked();
    render_with_footprint_mask();
    return;
  }

  tf2::Transform capture_to_global;
  capture_to_global.setIdentity();
  if (snapshot.header.frame_id != global_frame_) {
    try {
      const tf2::TimePoint capture_time{
        std::chrono::nanoseconds(snapshot.status.capture_timestamp_ns)};
      const geometry_msgs::msg::TransformStamped transform =
        tf_->lookupTransform(
        global_frame_, snapshot.header.frame_id, capture_time,
        tf2::durationFromSec(snapshot_transform_tolerance_sec_));
      tf2::fromMsg(transform.transform, capture_to_global);
    } catch (const tf2::TransformException & exception) {
      current_ = false;
      RCLCPP_WARN_THROTTLE(
        logger_, *clock_, 2000,
        "Obstacle snapshot capture-time transform %s -> %s unavailable: %s",
        snapshot.header.frame_id.c_str(), global_frame_.c_str(),
        exception.what());
      render_with_footprint_mask();
      return;
    }
  }

  std::set<WorldEvidenceCell> hit_cells;
  for (const Cell & cell : snapshot.cluster_accepted_cells) {
    if (static_cast<double>(cell.z_max) < min_obstacle_height_ ||
      static_cast<double>(cell.z_min) > max_obstacle_height_)
    {
      continue;
    }
    const RasterizedWorldCells raster =
      RasterizeWorldCellBounds(
      TransformSourceCell(snapshot, cell.ix, cell.iy, capture_to_global),
      evidence_resolution_m_, maximum_rasterized_cells_);
    if (!raster.valid ||
      raster.cells.size() >
      maximum_rasterized_cells_ - std::min(maximum_rasterized_cells_, hit_cells.size()))
    {
      invalidateSnapshotLocked();
      render_with_footprint_mask();
      return;
    }
    hit_cells.insert(raster.cells.begin(), raster.cells.end());
  }

  const std::vector<WorldEvidenceCell> hits(hit_cells.begin(), hit_cells.end());
  const bool new_generation = last_evidence_generation_ != 0U &&
    snapshot.status.run_generation != last_evidence_generation_;
  if (!temporal_evidence_->applyFrame(
      hits, unreported_cells_are_free_, new_generation))
  {
    invalidateSnapshotLocked();
    render_with_footprint_mask();
    return;
  }

  last_evidence_generation_ = snapshot.status.run_generation;
  last_valid_capture_ns_ = snapshot.status.capture_timestamp_ns;
  last_valid_status_ns_ = snapshot.status.status_timestamp_ns;
  latest_snapshot_.reset();
  render_with_footprint_mask();
  update_pending_ = true;
}

void ObstacleDecayLayer::renderTemporalEvidenceLocked()
{
  const unsigned char background = unreported_cells_are_free_ ? FREE_SPACE : NO_INFORMATION;
  std::fill(costmap_, costmap_ + size_x_ * size_y_, background);
  const double map_min_x = getOriginX();
  const double map_min_y = getOriginY();
  const double map_max_x = map_min_x + static_cast<double>(size_x_) * resolution_;
  const double map_max_y = map_min_y + static_cast<double>(size_y_) * resolution_;
  for (const WorldEvidenceCell & cell : temporal_evidence_->occupiedCells()) {
    const double tile_min_x = static_cast<double>(cell.x) * evidence_resolution_m_;
    const double tile_min_y = static_cast<double>(cell.y) * evidence_resolution_m_;
    const double tile_max_x = tile_min_x + evidence_resolution_m_;
    const double tile_max_y = tile_min_y + evidence_resolution_m_;
    if (tile_max_x < map_min_x || tile_max_y < map_min_y || tile_min_x > map_max_x ||
      tile_min_y > map_max_y)
    {
      continue;
    }
    const auto first_index = [this](const double minimum, const double map_minimum) {
        return static_cast<int>(std::floor((minimum - map_minimum) / resolution_));
      };
    const auto last_index = [this](const double maximum, const double map_minimum) {
        const double normalized = (maximum - map_minimum) / resolution_;
        const double tolerance =
          32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(normalized));
        return static_cast<int>(std::ceil(normalized - tolerance)) - 1;
      };
    int first_x = std::max(0, first_index(tile_min_x, map_min_x));
    int first_y = std::max(0, first_index(tile_min_y, map_min_y));
    int last_x = std::min(static_cast<int>(size_x_) - 1, last_index(tile_max_x, map_min_x));
    int last_y = std::min(static_cast<int>(size_y_) - 1, last_index(tile_max_y, map_min_y));
    for (int y = first_y; y <= last_y; ++y) {
      for (int x = first_x; x <= last_x; ++x) {
        costmap_[getIndex(static_cast<unsigned int>(x), static_cast<unsigned int>(y))] =
          LETHAL_OBSTACLE;
      }
    }
  }
}

bool ObstacleDecayLayer::snapshotCurrentLocked(const std::int64_t now_ns) const noexcept
{
  return snapshot_input_enabled_ && enabled_ && snapshot_active_ && current_ && now_ns > 0 &&
         last_valid_status_ns_ > 0 && last_valid_capture_ns_ > 0 &&
         !LaterThanTolerance(last_valid_status_ns_, now_ns, future_tolerance_ns_) &&
         !LaterThanTolerance(last_valid_capture_ns_, now_ns, future_tolerance_ns_) &&
         now_ns - last_valid_status_ns_ <= maximum_status_age_ns_ &&
         now_ns - last_valid_capture_ns_ <= maximum_capture_age_ns_;
}

}  // namespace nav2_costmap_2d
