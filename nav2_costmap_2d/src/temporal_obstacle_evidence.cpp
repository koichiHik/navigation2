// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include "nav2_costmap_2d/temporal_obstacle_evidence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace nav2_costmap_2d
{

RasterizedWorldCells RasterizeWorldCellBounds(
  const std::array<EvidencePoint2d, 4> & polygon,
  const double world_lattice_resolution,
  const std::size_t maximum_cells) noexcept
{
  RasterizedWorldCells result;
  try {
    if (!std::isfinite(world_lattice_resolution) || world_lattice_resolution <= 0.0 ||
      maximum_cells == 0U)
    {
      return result;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const EvidencePoint2d & point : polygon) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        return result;
      }
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
    }

    const double first_x_value = std::floor(min_x / world_lattice_resolution);
    const double first_y_value = std::floor(min_y / world_lattice_resolution);
    const double last_x_value = std::floor(max_x / world_lattice_resolution);
    const double last_y_value = std::floor(max_y / world_lattice_resolution);
    constexpr double kSafeIndexLimit =
      static_cast<double>(std::numeric_limits<std::int64_t>::max() / 4);
    if (!std::isfinite(first_x_value) || !std::isfinite(first_y_value) ||
      !std::isfinite(last_x_value) || !std::isfinite(last_y_value) ||
      first_x_value < -kSafeIndexLimit || first_y_value < -kSafeIndexLimit ||
      last_x_value > kSafeIndexLimit || last_y_value > kSafeIndexLimit)
    {
      return result;
    }

    const auto first_x = static_cast<std::int64_t>(first_x_value);
    const auto first_y = static_cast<std::int64_t>(first_y_value);
    const auto last_x = static_cast<std::int64_t>(last_x_value);
    const auto last_y = static_cast<std::int64_t>(last_y_value);
    const long double count_x =
      static_cast<long double>(last_x) - static_cast<long double>(first_x) + 1.0L;
    const long double count_y =
      static_cast<long double>(last_y) - static_cast<long double>(first_y) + 1.0L;
    if (count_x <= 0.0L || count_y <= 0.0L ||
      count_x * count_y > static_cast<long double>(maximum_cells))
    {
      return result;
    }

    result.cells.reserve(static_cast<std::size_t>(count_x * count_y));
    for (std::int64_t y = first_y; y <= last_y; ++y) {
      for (std::int64_t x = first_x; x <= last_x; ++x) {
        result.cells.push_back({x, y});
      }
    }
    result.valid = true;
  } catch (...) {
    result.cells.clear();
    result.valid = false;
  }
  return result;
}

bool TemporalEvidenceParameters::valid() const noexcept
{
  return hit_count_threshold > 0U &&
         hit_count_threshold <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         miss_decrement > 0 && clear_threshold >= 0 &&
         static_cast<std::size_t>(clear_threshold) < hit_count_threshold &&
         maximum_persistent_cells > 0U;
}

TemporalObstacleEvidence::TemporalObstacleEvidence(TemporalEvidenceParameters parameters)
: parameters_(std::move(parameters))
{
  if (!parameters_.valid()) {
    throw std::invalid_argument("invalid temporal obstacle evidence parameters");
  }
}

bool TemporalObstacleEvidence::applyFrame(
  const std::vector<WorldEvidenceCell> & hit_cells,
  const bool unreported_cells_are_free, const bool discard_unconfirmed) noexcept
{
  try {
    const std::set<WorldEvidenceCell> hits(hit_cells.begin(), hit_cells.end());
    std::map<WorldEvidenceCell, CellEvidence> next = evidence_;
    if (discard_unconfirmed) {
      for (auto iterator = next.begin(); iterator != next.end(); ) {
        if (!iterator->second.occupied) {
          iterator = next.erase(iterator);
        } else {
          ++iterator;
        }
      }
    }
    std::size_t new_cell_count = 0U;
    for (const WorldEvidenceCell & cell : hits) {
      if (next.count(cell) == 0U) {
        ++new_cell_count;
      }
    }
    if (new_cell_count > parameters_.maximum_persistent_cells -
      std::min(parameters_.maximum_persistent_cells, next.size()))
    {
      return false;
    }

    if (unreported_cells_are_free) {
      for (auto iterator = next.begin(); iterator != next.end(); ) {
        if (hits.count(iterator->first) != 0U) {
          ++iterator;
          continue;
        }
        CellEvidence & value = iterator->second;
        value.score = std::max(0, value.score - parameters_.miss_decrement);
        if (value.score <= parameters_.clear_threshold) {
          iterator = next.erase(iterator);
        } else {
          ++iterator;
        }
      }
    }

    const int hit_limit = static_cast<int>(parameters_.hit_count_threshold);
    for (const WorldEvidenceCell & cell : hits) {
      CellEvidence & value = next[cell];
      value.score = std::min(hit_limit, value.score + 1);
      if (value.score >= hit_limit) {
        value.occupied = true;
      }
    }

    evidence_ = std::move(next);
    return true;
  } catch (...) {
    return false;
  }
}

std::vector<WorldEvidenceCell> TemporalObstacleEvidence::occupiedCells() const
{
  std::vector<WorldEvidenceCell> result;
  result.reserve(occupiedCellCount());
  for (const auto & item : evidence_) {
    if (item.second.occupied) {
      result.push_back(item.first);
    }
  }
  return result;
}

std::optional<int> TemporalObstacleEvidence::score(const WorldEvidenceCell & cell) const noexcept
{
  const auto found = evidence_.find(cell);
  return found == evidence_.end() ? std::nullopt : std::optional<int>{found->second.score};
}

std::size_t TemporalObstacleEvidence::persistentCellCount() const noexcept
{
  return evidence_.size();
}

std::size_t TemporalObstacleEvidence::occupiedCellCount() const noexcept
{
  return static_cast<std::size_t>(std::count_if(
           evidence_.begin(), evidence_.end(),
           [](const auto & item) {return item.second.occupied;}));
}

void TemporalObstacleEvidence::reset() noexcept {evidence_.clear();}

}  // namespace nav2_costmap_2d
