// Copyright (c) 2026 koichiHik
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include "nav2_mppi_controller/critic_function.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace mppi::critics
{

class ThrowingTestCritic : public CriticFunction
{
public:
  void initialize() override
  {
    auto getParam = parameters_handler_->getParamGetter(name_);
    getParam(weight_, "cost_weight", 1.0);
    throw std::runtime_error("test critic configuration failed");
  }

  void score(CriticData &) override {}

private:
  float weight_{0};
};

}  // namespace mppi::critics

PLUGINLIB_EXPORT_CLASS(mppi::critics::ThrowingTestCritic, mppi::critics::CriticFunction)
