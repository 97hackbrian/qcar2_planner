cmake_minimum_required(VERSION 3.8)
project(qcar2_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===========================================================================
# Dependencies
# ===========================================================================
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# ===========================================================================
# Service generation  (GetDirectionalPath.srv)
# rosidl automatically generates Python bindings as 'qcar2_planner.srv'
# ===========================================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDirectionalPath.srv"
  DEPENDENCIES geometry_msgs nav_msgs
)

# ===========================================================================
# Install Python executable scripts
# ===========================================================================
install(PROGRAMS
  scripts/map_processor_node.py
  scripts/exploration_manager_node.py
  scripts/directional_planner_server.py
  scripts/map_loader_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===========================================================================
# Install launch & config
# ===========================================================================
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# ===========================================================================
# Testing
# ===========================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
