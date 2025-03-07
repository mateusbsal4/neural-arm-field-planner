# Trajectory planning in Genesis

This repository consists of a ROS workspace with scripts and utilities for benchmarking the Predictive Multi Agent Framework ([PMAF](https://github.com/riddhiman13/multi_agent_vector_fields/tree/8f151309d020fc34493607b42d175da027354e84)) and offline sampling-based trajectory planners in simulated cluttered environments with the Franka Emika Panda manipulator in the Genesis physics engine.

## Dependencies   
- [Genesis](https://genesis-world.readthedocs.io/en/latest/index.html)
- OMPL
- ROS Noetic


## Repository Structure
```
trajectory_planning_genesis
├── src
│   └── genesis_inverse_kinematics
│       └── launch
│       └── models
│       └── scripts/
│           └── IK_ompl.py
│           └── IK_pmaf.py
│       └── src/genesis-inverse-kinematics
│       └── CMakeLists.txt
│       └── package.xml
│       └── setup.py
│   └── multi_agent_vector_fields
└── README.md
```

## Runnning the planners
### 1. RRT-Connect
The script `IK_ompl.py` implements inverse kinematics and motion planning in an offline fashion using OMPL´s RRT-Connect trajectory planner.
- **How to Run:**
  Navigate to the root of the wokspace and run:
  ```bash
  python src/genesis_inverse_kinematics/scripts/IK_ompl.py
  ```
-  [Example experiment](src/genesis_inverse_kinematics/videos/ompl.mp4)


### 2. PMAF  
  The `IK_pmaf.py` ROS node publishes the target and current positions of the robot’s TCP and executes to the trajectory waypoints generated in real time by the PMAF planner node.
- **How to Run:**
 Navigate to the root of the ROS workspace, then build it:
  ```bash
  catkin_make
  ```
  Source the workspace 
  ```bash
  source devel/setup.bash
  ```
  Launch the Genesis node:
  ```bash
  roslaunch genesis_inverse_kinematics ik_genesis.launch
  ```
  Open another terminal, source the ROS workspace, and launch the PMAF planner:
  ```bash
  roslaunch multi_agent_vector_fields main_demo.launch
  ```
-  [Example experiment](src/genesis_inverse_kinematics/videos/pmaf.mp4)  

  
  


  
