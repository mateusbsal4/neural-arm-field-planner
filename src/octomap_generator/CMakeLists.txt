cmake_minimum_required(VERSION 3.0.2)
project(octomap_generator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/publish_depth_image.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)