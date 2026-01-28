cmake_minimum_required(VERSION 3.8)
project(lidar_cpp_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(PCL REQUIRED COMPONENTS common filters segmentation search)

include_directories(${PCL_INCLUDE_DIRS})

add_executable(cloud_monitor src/cloud_monitor.cpp)
ament_target_dependencies(cloud_monitor rclcpp sensor_msgs)

add_executable(cloud_filter src/cloud_filter.cpp)
ament_target_dependencies(cloud_filter rclcpp sensor_msgs)

add_executable(object_detector_pcl src/object_detector_pcl.cpp)
ament_target_dependencies(object_detector_pcl
  rclcpp
  sensor_msgs
  visualization_msgs
  std_msgs
  pcl_conversions
  tf2_ros
  tf2_sensor_msgs
  geometry_msgs
)
target_link_libraries(object_detector_pcl ${PCL_LIBRARIES})

install(TARGETS
  cloud_monitor
  cloud_filter
  object_detector_pcl
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
