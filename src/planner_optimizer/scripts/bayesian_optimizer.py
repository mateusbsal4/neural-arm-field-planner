#!/usr/bin/env python3
import rospy
import roslaunch
import yaml
import time
import pandas as pd
import numpy as np
import os
import sys
sys.path.append("/home/geriatronics/miniconda3/envs/ros_perception/lib/python3.9/site-packages")
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from std_msgs.msg import Float32

global_cost = None

def cost_callback(msg):
    global global_cost
    global_cost = msg.data

def launch_experiment():
    """
    Launch the IK_pmaf node and the planner node using their launch files.
    The planner node uses the temporary config file (temp_yaml_file) for its parameters
    """
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    # Launch IK node
    ik_launch_file = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch"
    ik_parent = roslaunch.parent.ROSLaunchParent(uuid, [ik_launch_file])
    ik_parent.start()
    rospy.loginfo("IK node launched.")
    # Wait a short time to ensure the IK node is running
    time.sleep(5)
    # Launch planner node
    planner_launch_file = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/launch/main_demo.launch"
    planner_parent = roslaunch.parent.ROSLaunchParent(uuid, [planner_launch_file])
    planner_parent.start()
    rospy.loginfo("Planner node launched.")
    return ik_parent, planner_parent

def shutdown_experiment(ik_parent, planner_parent):
    ik_parent.shutdown()
    planner_parent.shutdown()
    rospy.loginfo("Experiment nodes shutdown.")

def run_experiment():
    """
    Launch the experiment nodes with the current parameters,
    wait (up to 60s) for a cost message on /cost, then return the cost.
    """
    global global_cost
    global_cost = None
    ik_parent, planner_parent = launch_experiment()
    # Subscribe (once per trial) to the /cost topic
    cost_sub = rospy.Subscriber("/cost", Float32, cost_callback)
    wait_time = 0
    while global_cost is None and wait_time < 60:
        time.sleep(1)
        wait_time += 1
    cost_val = global_cost 
    rospy.loginfo("Obtained cost: {:.2f}".format(cost_val))
    shutdown_experiment(ik_parent, planner_parent)
    # Give time for nodes to shutdown completely
    time.sleep(5)
    return cost_val

if __name__ == "__main__":
    rospy.init_node("bayes_optimizer_node", anonymous=True)

    # Define a HEBO design space for 36 parameters:
    #  – 1 scalar: detect_shell_rad
    #  – 5 vectors each of length 7: k_a_ee, k_c_ee, k_r_ee, k_d_ee, k_manip
    design_list = []
    design_list.append({'name': 'detect_shell_rad', 'type': 'num', 'lb': 0.25, 'ub': 0.75})
    for grp, (name, lb, ub) in enumerate([
        ('k_a_ee', 1.0, 5.0),
        ('k_c_ee', 1.0, 5.0),
        ('k_r_ee', 1.0, 5.0),
        ('k_d_ee', 1.0, 5.0),
        ('k_manip', 1.0, 5.0)
    ]):
        for j in range(7):
            design_list.append({'name': f'{name}_{j}', 'type': 'num', 'lb': lb, 'ub': ub})
    space = DesignSpace().parse(design_list)
    hebo_batch = HEBO(space, model_name='gp', rand_sample=4)
    # Temporary config file path 
    temp_yaml_file = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
    num_iterations = 16
    for i in range(num_iterations):
        rec = hebo_batch.suggest(n_suggestions=1)  
        rospy.loginfo("Iteration {}: Suggested parameters batch:".format(i))
        rospy.loginfo("\n{}".format(rec))
        # Convert the suggested parameters (a DataFrame) to a dictionary.
        rec_dict = rec.to_dict(orient='records')[0]
        # Build the parameter dictionary for agent_parameters.yaml.
        param_dict = {
            'detect_shell_rad': rec_dict['detect_shell_rad'],
            'agent_mass': 1.0,
            'agent_radius': 0.2,
            'velocity_max': 0.5,
            'approach_dist': 0.25,
            'k_a_ee': [rec_dict[f'k_a_ee_{j}'] for j in range(7)],
            'k_c_ee': [rec_dict[f'k_c_ee_{j}'] for j in range(7)],
            'k_r_ee': [rec_dict[f'k_r_ee_{j}'] for j in range(7)],
            'k_r_force': [0.0]*7,  # fixed at 0
            'k_d_ee': [rec_dict[f'k_d_ee_{j}'] for j in range(7)],
            'k_manip': [rec_dict[f'k_manip_{j}'] for j in range(7)]
        }
        # Write the parameters to the temporary YAML file
        with open(temp_yaml_file, 'w') as f:
            yaml.dump(param_dict, f)
        rospy.loginfo("Written temporary agent parameters for suggestion {}:".format(i))
        rospy.loginfo(param_dict)
        # Run the experiment (launch IK and planner nodes, wait for cost)
        cost = run_experiment()
        rospy.loginfo("Suggestion {}: Obtained cost = {:.2f}".format(i, cost))
        # Observe the result with HEBO
        hebo_batch.observe(rec, np.array([[cost]]))
        rospy.loginfo("After iteration {}, best cost so far = {:.2f}".format(i, hebo_batch.y.min()))
    rospy.loginfo("Optimization complete. Best parameters found:")
    rospy.loginfo(hebo_batch.rec)