# Trajectory planning in Genesis

This repository consists of a ROS package with scripts and utilities for benchmarking trajectory planners in simulated cluttered environments with the Franka Emika Panda manipulator in the Genesis physics engine.

## Dependencies   
- [Genesis](https://genesis-world.readthedocs.io/en/latest/index.html)
- OMPL
- ROS Noetic
- [PMAF Planner](https://github.com/riddhiman13/multi_agent_vector_fields/tree/8f151309d020fc34493607b42d175da027354e84)

## Repository Structure
```
genesis-inverse-kinematics
├── launch
│   └── ik_genesis.launch
├── models/
│   └── YCB
│   └── franka_emika_panda
├── scripts/
│   └── IK_ompl.py
│   └── IK_pmaf.py
├── src/genesis-inverse-kinematics
│   └── evaluate_path.py
│   └── task_setup.py
├── CMakeLists.txt
├── package.xml
└── setup.py
```

### 1. `IK_ompl.py`
- **Purpose:**  
  Implements inverse kinematics and motion planning using OMPL´s RRT-Connect trajectory planner. 
- **How to Run:**  
  ```bash
  python scripts/IK_ompl.py
-  **Example output:**
https://github.com/user-attachments/assets/06c9f5a3-67f4-4f3e-af35-4217792a989a


### 2. `IK_pmaf.py`
- **Purpose:**  
  A ROS node that defines a target point, subscribes to the trajectory waypoints published by the PMAF planner, and executes them in the Genesis environment.
- **How to Run:**
  Navigate to the root of the ROS wokspace, build it and launch the ROS node with
  ```bash
  roslaunch genesis_inverse_kinematics ik_genesis.launch
  ```
  Launch the PMAF planner by following the corresponding [instructions](https://github.com/riddhiman13/multi_agent_vector_fields/blob/8f151309d020fc34493607b42d175da027354e84/README.md)
-  **Example output:**


  
  


  
