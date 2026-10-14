// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include "nav2_costmap_2d/temporal_obstacle_evidence.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nav2_costmap_2d
{
namespace
{

TEST(TemporalObstacleEvidence, RequiresConfiguredNumberOfHitFrames) {
  TemporalEvidenceParameters parameters;
  parameters.hit_count_threshold = 3U;
  TemporalObstacleEvidence evidence(parameters);
  const WorldEvidenceCell cell{10, 20};

  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  EXPECT_TRUE(evidence.occupiedCells().empty());
  EXPECT_EQ(evidence.score(cell), 1);
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  EXPECT_TRUE(evidence.occupiedCells().empty());
  EXPECT_EQ(evidence.score(cell), 2);
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  EXPECT_EQ(evidence.occupiedCells(), std::vector<WorldEvidenceCell>({cell}));
  EXPECT_EQ(evidence.score(cell), 3);

  // Scores saturate at the hit threshold; max_evidence is intentionally not
  // another tuning parameter in the simplified policy.
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  EXPECT_EQ(evidence.score(cell), 3);
}

TEST(TemporalObstacleEvidence, ValidMissUsesHysteresisBeforeClearing) {
  TemporalObstacleEvidence evidence(TemporalEvidenceParameters{});
  const WorldEvidenceCell cell{1, 2};
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  ASSERT_EQ(evidence.occupiedCellCount(), 1U);

  ASSERT_TRUE(evidence.applyFrame({}, true));
  EXPECT_EQ(evidence.occupiedCellCount(), 1U);
  EXPECT_EQ(evidence.score(cell), 1);
  ASSERT_TRUE(evidence.applyFrame({}, true));
  EXPECT_EQ(evidence.occupiedCellCount(), 0U);
  EXPECT_FALSE(evidence.score(cell).has_value());
}

TEST(TemporalObstacleEvidence, MissingCoverageDoesNotImplicitlyClear) {
  TemporalObstacleEvidence evidence(TemporalEvidenceParameters{});
  const WorldEvidenceCell cell{1, 2};
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  ASSERT_TRUE(evidence.applyFrame({cell}, true));
  ASSERT_TRUE(evidence.applyFrame({}, false));
  ASSERT_TRUE(evidence.applyFrame({}, false));
  EXPECT_EQ(evidence.occupiedCellCount(), 1U);
  EXPECT_EQ(evidence.score(cell), 2);
}

TEST(TemporalObstacleEvidence, DuplicateCellsInOneFrameIncrementOnce) {
  TemporalObstacleEvidence evidence(TemporalEvidenceParameters{});
  const WorldEvidenceCell cell{3, 4};
  ASSERT_TRUE(evidence.applyFrame({cell, cell, cell}, true));
  EXPECT_EQ(evidence.score(cell), 1);
  EXPECT_EQ(evidence.occupiedCellCount(), 0U);
}

TEST(TemporalObstacleEvidence, NewGenerationDropsOnlyUnconfirmedEvidence) {
  TemporalObstacleEvidence evidence(TemporalEvidenceParameters{});
  const WorldEvidenceCell confirmed{1, 1};
  const WorldEvidenceCell pending{2, 2};
  ASSERT_TRUE(evidence.applyFrame({confirmed}, false));
  ASSERT_TRUE(evidence.applyFrame({confirmed, pending}, false));
  ASSERT_EQ(evidence.occupiedCellCount(), 1U);
  ASSERT_EQ(evidence.score(pending), 1);

  ASSERT_TRUE(evidence.applyFrame({}, false, true));
  EXPECT_EQ(evidence.occupiedCells(), std::vector<WorldEvidenceCell>({confirmed}));
  EXPECT_EQ(evidence.score(confirmed), 2);
  EXPECT_FALSE(evidence.score(pending).has_value());
}

TEST(TemporalObstacleEvidence, CapacityFailureIsTransactional) {
  TemporalEvidenceParameters parameters;
  parameters.maximum_persistent_cells = 1U;
  TemporalObstacleEvidence evidence(parameters);
  const WorldEvidenceCell retained{1, 1};
  ASSERT_TRUE(evidence.applyFrame({retained}, true));
  EXPECT_FALSE(evidence.applyFrame({retained, {2, 2}}, true));
  EXPECT_EQ(evidence.persistentCellCount(), 1U);
  EXPECT_EQ(evidence.score(retained), 1);
  EXPECT_FALSE(evidence.score({2, 2}).has_value());
}

TEST(TemporalObstacleEvidence, RejectsInvalidParameters) {
  TemporalEvidenceParameters parameters;
  parameters.hit_count_threshold = 0U;
  EXPECT_THROW(TemporalObstacleEvidence{parameters}, std::invalid_argument);
  parameters.hit_count_threshold = 2U;
  parameters.clear_threshold = 2;
  EXPECT_THROW(TemporalObstacleEvidence{parameters}, std::invalid_argument);
  parameters.clear_threshold = 0;
  parameters.miss_decrement = 0;
  EXPECT_THROW(TemporalObstacleEvidence{parameters}, std::invalid_argument);
}

TEST(TemporalObstacleEvidence, RasterizesConservativeWorldBounds) {
  const std::array<EvidencePoint2d, 4> polygon{
    {{1.01, 2.01}, {1.19, 2.01}, {1.19, 2.19}, {1.01, 2.19}}};
  const RasterizedWorldCells result = RasterizeWorldCellBounds(polygon, 0.1, 100U);
  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.cells.size(), 4U);
  EXPECT_NE(
    std::find(result.cells.begin(), result.cells.end(), WorldEvidenceCell{10, 20}),
    result.cells.end());
  EXPECT_NE(
    std::find(result.cells.begin(), result.cells.end(), WorldEvidenceCell{11, 21}),
    result.cells.end());
}

TEST(TemporalObstacleEvidence, RasterizationRejectsNonFiniteOrExcessiveInput) {
  std::array<EvidencePoint2d, 4> polygon{{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
  EXPECT_FALSE(RasterizeWorldCellBounds(polygon, 0.1, 10U).valid);
  polygon[0].x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(RasterizeWorldCellBounds(polygon, 0.1, 1000U).valid);
}

}  // namespace
}  // namespace nav2_costmap_2d
