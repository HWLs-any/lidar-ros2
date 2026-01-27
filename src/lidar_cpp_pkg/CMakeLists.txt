cmake_minimum_required(VERSION 3.8)
project(lidar_cpp_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C++17 (good for ROS 2 Jazzy)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(cloud_monitor src/cloud_monitor.cpp)
ament_target_dependencies(cloud_monitor rclcpp sensor_msgs)

add_executable(cloud_filter src/cloud_filter.cpp)
ament_target_dependencies(cloud_filter rclcpp sensor_msgs)

install(TARGETS
  cloud_monitor
  cloud_filter
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
