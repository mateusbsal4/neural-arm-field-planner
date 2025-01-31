cmake_minimum_required(VERSION 3.0.2)
project(genesis_inverse_kinematics)

# Find required catkin components
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
)

# Declare the catkin package
catkin_package()

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/IK_genesis.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
