# Machine-Learning Based Robot Motion Planning for the Franka Emika Panda Robot

This repository contains the codebase for a research project from the Technical University of Munich focused on applying machine learning techniques to the problem of robot motion planning. The primary goal is to develop a system that can generalize motion planning strategies across a variety of environments by learning optimal planner parameters.

The project leverages the Robot Operating System (ROS) and the Genesis robotics simulator. The core of the project is a pipeline that:
1.  Perceives a scene using a simulated depth camera.
2.  Constructs a world representation (voxel grid) from the sensor data.
3.  Uses a vector-field-based planner (`multi_agent_vector_fields`) to generate a trajectory for the Franka Emika Panda robot arm.
4.  Employs a Bayesian Optimization framework (HEBO) to find the optimal hyperparameters for the planner in a given environment.
5.  Utilizes a PointNet-based deep learning model to learn the mapping from an environment's features to the optimal planner parameters.

## Codebase Structure

The project is organized into several ROS packages located in the `src/` directory:

-   `src/apriltag` & `src/apriltag_ros`: Contains the core library and ROS wrapper for AprilTag detection, used for camera and robot localization and calibration.
-   `src/dataset_generator`: Includes scripts and tools to systematically generate datasets for training the machine learning models. This involves creating various scenes in the Genesis simulator and recording the corresponding sensor data and planner performance.
-   `src/genesis_inverse_kinematics`: Handles robot motion, containing both the inverse kinematics (IK) solver for the Genesis simulator (`scripts/IK_pmaf.py`) and the controller (`scripts/robot_controller.py`) for executing trajectories on the physical Franka Emika Panda robot.
-   `src/multi_agent_vector_fields`: An implementation of a multi-agent navigation system using vector fields for motion planning. This package is responsible for generating the robot's trajectory while avoiding obstacles.
-   `src/percept`: The perception pipeline. This package processes depth camera data from the simulator, transforms it into a point cloud, and generates a voxel grid representation of the environment for the planner.
-   `src/planner_optimizer`: Contains the implementation of the Bayesian Optimization using HEBO. This package is used to tune the hyperparameters of the `multi_agent_vector_fields` planner to optimize a cost functional (e.g., path clearance, smoothness, length).
-   `src/point_net`: Implements a PointNet-based deep learning model. This is used to learn a direct mapping from a perceived environment (as a point cloud) to the optimal planner parameters, allowing the system to bypass the expensive optimization process in new environments.

## End-to-End Pipeline Instructions

The following instructions describe how to set up the environment, build the ROS packages, and run the final end-to-end pipeline.

### Environment Setup

This project requires two distinct Conda environments: `genesis_ros` and `ros_perception`.

You can create them using the provided YAML files:

```bash
# Create the genesis_ros environment
conda env create -f env/genesis_ros_env.yaml

# Create the ros_perception environment
conda env create -f env/ros_perception_env.yaml
```

### Build Instructions

After activating the correct Conda environment, you need to build the ROS packages. You will need two separate terminals.

**Terminal 1 (activate `genesis_ros`):**
```bash
conda activate genesis_ros
# Navigate to the root of your ROS workspace
catkin_make -DCATKIN_WHITELIST_PACKAGES=genesis_inverse_kinematics
catkin_make -DCATKIN_WHITELIST_PACKAGES=multi_agent_vector_fields
```

**Terminal 2 (activate `ros_perception`):**
```bash
conda activate ros_perception
# Navigate to the root of your ROS workspace
catkin_make -DCATKIN_WHITELIST_PACKAGES=percept
catkin_make -DCATKIN_WHITELIST_PACKAGES=point_net
```

### Running the Pipeline

Finally, to run the pipeline, execute the following commands.

**Terminal 2 (with `ros_perception` activated):**
```bash
conda activate ros_perception
# Source the workspace
source devel/setup.bash

# Launch the pipeline
roslaunch point_net real_robot_pipeline.launch
```
This will launch the complete perception and planning pipeline, which listens for goals, perceives the environment, and moves the robot accordingly using the learned model.

## Results

The following animations demonstrate the performance of the final system in two distinct scenarios.

### Horizontal Obstacle Avoidance
![Horizontal Obstacle Avoidance](./results/result_horizontal.gif)

### Vertical Obstacle Avoidance
![Vertical Obstacle Avoidance](./results/result_vertical.gif)

For comparison, the following animations show the planner's performance using fixed gains, based on the original [PMAF](https://github.com/riddhiman13/multi_agent_vector_fields).

### Horizontal Obstacle Avoidance (Fixed Gains)
![Horizontal Obstacle Avoidance (Fixed Gains)](./results/planner_without_gain_inference_horizontal.gif)

### Vertical Obstacle Avoidance (Fixed Gains)
![Vertical Obstacle Avoidance (Fixed Gains)](./results/planner_without_gain_inference_vertical.gif)


