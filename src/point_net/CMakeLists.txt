cmake_minimum_required(VERSION 3.0.2)
project(point_net)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

# Setup Python packaging
catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/full_pipeline.py
  scripts/real_robot_pipeline.py
  src/point_net/model.py
  src/point_net/infer_gains.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install your Python modules from src
install(DIRECTORY src/point_net/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
