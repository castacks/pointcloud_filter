cmake_minimum_required(VERSION 3.8)
project(pointcloud_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation kdtree)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Outlier Filter Node
add_executable(outlier_filter_node 
  src/outlier_filter_node.cpp
)

ament_target_dependencies(outlier_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(outlier_filter_node
  ${PCL_LIBRARIES}
)

# Install executables
install(TARGETS
  outlier_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
