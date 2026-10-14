// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef NAV2_COSTMAP_2D__TEMPORAL_OBSTACLE_EVIDENCE_HPP_
#define NAV2_COSTMAP_2D__TEMPORAL_OBSTACLE_EVIDENCE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace nav2_costmap_2d
{

struct EvidencePoint2d
{
  double x{0.0};
  double y{0.0};
};

struct WorldEvidenceCell
{
  std::int64_t x{0};
  std::int64_t y{0};

  bool operator<(const WorldEvidenceCell & other) const noexcept
  {
    return x < other.x || (x == other.x && y < other.y);
  }

  bool operator==(const WorldEvidenceCell & other) const noexcept
  {
    return x == other.x && y == other.y;
  }
};

struct RasterizedWorldCells
{
  bool valid{false};
  std::vector<WorldEvidenceCell> cells;
};

// Conservatively covers the transformed source-cell polygon with a fixed-world
// axis-aligned lattice. The small AABB over-approximation is intentional: a
// measured obstacle may occupy a little more Costmap area, but never less.
[[nodiscard]] RasterizedWorldCells RasterizeWorldCellBounds(
  const std::array<EvidencePoint2d, 4> & polygon, double world_lattice_resolution,
  std::size_t maximum_cells) noexcept;

struct TemporalEvidenceParameters
{
  std::size_t hit_count_threshold{2U};
  int miss_decrement{1};
  int clear_threshold{0};
  std::size_t maximum_persistent_cells{500'000U};

  [[nodiscard]] bool valid() const noexcept;
};

// Minimal temporal filter for an accepted-only obstacle stream. Each call is
// one distinct, valid source frame. Hits increment a fixed-world cell once;
// omitted cells decrement only when the producer explicitly operates in the
// HITS_ONLY "unreported is free" mode. Once promoted, a cell remains occupied
// until its score reaches clear_threshold.
class TemporalObstacleEvidence final
{
public:
  explicit TemporalObstacleEvidence(TemporalEvidenceParameters parameters);

  [[nodiscard]] bool applyFrame(
    const std::vector<WorldEvidenceCell> & hit_cells,
    bool unreported_cells_are_free, bool discard_unconfirmed = false) noexcept;
  [[nodiscard]] std::vector<WorldEvidenceCell> occupiedCells() const;
  [[nodiscard]] std::optional<int> score(const WorldEvidenceCell & cell) const noexcept;
  [[nodiscard]] std::size_t persistentCellCount() const noexcept;
  [[nodiscard]] std::size_t occupiedCellCount() const noexcept;
  void reset() noexcept;

private:
  struct CellEvidence
  {
    int score{0};
    bool occupied{false};
  };

  TemporalEvidenceParameters parameters_;
  std::map<WorldEvidenceCell, CellEvidence> evidence_;
};

}  // namespace nav2_costmap_2d

#endif  // NAV2_COSTMAP_2D__TEMPORAL_OBSTACLE_EVIDENCE_HPP_
