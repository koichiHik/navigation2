// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <pluginlib/class_loader.hpp>
#include <string>
#include <vector>

#include "nav2_costmap_2d/layer.hpp"

TEST(CostmapPluginsTester, obstacleDecayLoadsThroughPluginlib) {
  pluginlib::ClassLoader<nav2_costmap_2d::Layer> loader("nav2_costmap_2d",
    "nav2_costmap_2d::Layer");
  const std::vector<std::string> declared = loader.getDeclaredClasses();
  ASSERT_NE(
    std::find(declared.begin(), declared.end(), "nav2_costmap_2d::ObstacleDecayLayer"),
    declared.end());

  const std::shared_ptr<nav2_costmap_2d::Layer> layer =
    loader.createSharedInstance("nav2_costmap_2d::ObstacleDecayLayer");
  ASSERT_NE(layer, nullptr);
}
