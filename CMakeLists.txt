cmake_minimum_required(VERSION 3.0.2)
project(trajectory_planning_genesis)

# Find required catkin components
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
)

# Declare the catkin package
catkin_package()

# Setup Python packaging
catkin_python_setup()

# Install Python scripts (executable nodes)
catkin_install_python(PROGRAMS
  scripts/IK_pmaf.py
  scripts/multilink_IK_pmaf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install your Python modules from src
install(DIRECTORY src/trajectory_planning_genesis/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
