set(TEST_NAMES
  optimizer_smoke_test
  controller_state_transition_test
  models_test
  noise_generator_test
  parameter_handler_test
  motion_model_tests
  trajectory_visualizer_tests
  utils_test
  path_handler_test
  critic_manager_test
  optimizer_unit_tests
)

foreach(name IN LISTS TEST_NAMES)
  ament_add_gtest(${name}
    ${name}.cpp
  )

  ament_target_dependencies(${name}
    ${dependencies_pkgs}
  )

  target_link_libraries(${name}
    mppi_controller
  )

  if(${TEST_DEBUG_INFO})
    target_compile_definitions(${name} PUBLIC -DTEST_DEBUG_INFO)
  endif()

endforeach()

# This is a special case requiring linking against the critics library
ament_add_gtest(critics_tests critics_tests.cpp)
ament_target_dependencies(critics_tests ${dependencies_pkgs})
target_link_libraries(critics_tests mppi_controller mppi_critics)
if(${TEST_DEBUG_INFO})
  target_compile_definitions(critics_tests PUBLIC -DTEST_DEBUG_INFO)
endif()

# The manifest is test-private; the plugin is built only with BUILD_TESTING.
add_library(mppi_failing_test_critic SHARED failing_test_critic.cpp)
ament_target_dependencies(mppi_failing_test_critic ${dependencies_pkgs})
target_link_libraries(mppi_failing_test_critic mppi_controller)
install(TARGETS mppi_failing_test_critic LIBRARY DESTINATION lib COMPONENT tests)
configure_file(failing_test_critic.xml.in failing_test_critic.xml @ONLY)
configure_file(../package.xml package.xml COPYONLY)
target_compile_definitions(critic_manager_test PRIVATE
  MPPI_FAILING_CRITIC_XML="${CMAKE_CURRENT_BINARY_DIR}/failing_test_critic.xml")
add_dependencies(critic_manager_test mppi_failing_test_critic)
