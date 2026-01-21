cmake_minimum_required(VERSION 2.8.3)
project(plan_manager)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  visualization_msgs
  uneven_map
  front_end
  back_end
  mpc_controller
  sensor_msgs
  pcl_ros
  pcl_conversions
  message_generation
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  PointCloudToGrid.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_manager
 CATKIN_DEPENDS roscpp std_msgs uneven_map visualization_msgs front_end back_end mpc_controller sensor_msgs message_runtime
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    /opt/ros/noetic/include/ompl-1.6
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(plan_manager
    src/plan_manager.cpp
)

add_library(only_planner_lib
    src/only_planner.cpp
)

target_link_libraries(plan_manager
    ${catkin_LIBRARIES}
    /opt/ros/noetic/lib/x86_64-linux-gnu/libompl.so
)

target_link_libraries(only_planner_lib
    ${catkin_LIBRARIES}
    /opt/ros/noetic/lib/x86_64-linux-gnu/libompl.so
)

add_executable(manager_node src/manager_node.cpp )
add_executable(only_planner_node src/only_planner_node.cpp )

# PointCloud to Grid Converter
add_library(pointcloud_to_grid_converter
    src/pointcloud_to_grid_converter.cpp
)

add_executable(pointcloud_to_grid_converter_node src/pointcloud_to_grid_converter_node.cpp)

target_link_libraries(manager_node
    plan_manager
    ${catkin_LIBRARIES}
    /opt/ros/noetic/lib/x86_64-linux-gnu/libompl.so
)

target_link_libraries(only_planner_node
    only_planner_lib
    ${catkin_LIBRARIES}
    /opt/ros/noetic/lib/x86_64-linux-gnu/libompl.so
)

target_link_libraries(pointcloud_to_grid_converter
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

target_link_libraries(pointcloud_to_grid_converter_node
    pointcloud_to_grid_converter
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

# Install Python scripts
catkin_install_python(PROGRAMS
    scripts/data_generate.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
