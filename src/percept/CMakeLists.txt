cmake_minimum_required(VERSION 3.0.2)
project(percept)

find_package(catkin REQUIRED COMPONENTS
  rosbag
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  message_filters
  pcl_conversions
  pcl_ros
  realsense2_camera
)

# Find Python from Conda environment
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env python -c
    "import sys; print(sys.executable)"
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the Python executable
set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
message(STATUS "Using Python from: ${PYTHON_EXECUTABLE}")

# Set the Python include directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
    "from sysconfig import get_paths as gp; print(gp()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYTHON_INCLUDE_DIR})

# Set the Python library directory
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
    "from sysconfig import get_paths as gp; print(gp()['stdlib'])"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories(${PYTHON_LIB_DIR})

# Declare the catkin package
catkin_package()

# Setup Python packaging
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  src/perception_node.py
  src/perception_pipeline.py
  src/sim_pipeline.py
  src/static_depth_publisher.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(DIRECTORY 
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
