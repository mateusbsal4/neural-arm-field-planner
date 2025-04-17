#!/usr/bin/env python3
import rospy
import roslaunch
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from std_msgs.msg import Float32MultiArray

sys.path.append("/home/geriatronics/miniconda3/envs/ros_perception/lib/python3.9/site-packages")
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

# Global variables to store costs
global_cost = None
individual_costs = None

def cost_callback(msg):
    """
    Callback to handle the received list of costs and compute the total cost.
    """
    global global_cost, individual_costs
    individual_costs = msg.data  # Store the list of costs
    global_cost = sum(individual_costs)  # Compute the total cost as the sum of individual costs

def launch_experiment():
    """
    Launch the IK_pmaf node and the planner node using their launch files.
    """
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    # Launch IK node
    ik_launch_file = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch"
    ik_parent = roslaunch.parent.ROSLaunchParent(uuid, [ik_launch_file])
    ik_parent.start()
    rospy.loginfo("IK node launched.")
    time.sleep(5)  # Wait to ensure the IK node is running
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
    Launch the experiment nodes, wait for a cost message, and return the total cost.
    """
    global global_cost, individual_costs
    global_cost = None
    individual_costs = None
    ik_parent, planner_parent = launch_experiment()
    cost_sub = rospy.Subscriber("/cost", Float32MultiArray, cost_callback)
    wait_time = 0
    while global_cost is None and wait_time < 60:
        time.sleep(1)
        wait_time += 1
    total_cost = global_cost
    rospy.loginfo("Obtained total cost: {:.2f}".format(total_cost))
    rospy.loginfo("Individual costs: {}".format(individual_costs))
    shutdown_experiment(ik_parent, planner_parent)
    time.sleep(5)  # Allow nodes to shut down completely
    return total_cost, individual_costs

if __name__ == "__main__":
    rospy.init_node("bayes_optimizer_node", anonymous=True)

    # Define a HEBO design space
    design_list = [{'name': 'detect_shell_rad', 'type': 'num', 'lb': 0.25, 'ub': 0.75}]
    for name, lb, ub in [
        ('k_a_ee', 1.0, 5.0),
        ('k_c_ee', 1.0, 5.0),
        ('k_r_ee', 1.0, 5.0),
        ('k_d_ee', 1.0, 5.0),
        ('k_manip', 1.0, 5.0)
    ]:
        for j in range(7):
            design_list.append({'name': f'{name}_{j}', 'type': 'num', 'lb': lb, 'ub': ub})
    space = DesignSpace().parse(design_list)
    #cfg = {
    #    "lr": 0.001,
    #    "num_epochs": 100,
    #    "verbose": False,
    #    "noise_lb": 8e-4,
    #    "pred_likeli": False,
    #}
    #hebo_batch = HEBO(space, model_name='gp', rand_sample=4, model_config=cfg)
    hebo_batch = HEBO(space, model_name='svgp', rand_sample=4)  
    # Temporary config file path
    temp_yaml_file = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
    num_iterations = 25

    # Initialize cost tracking
    costs = []
    individual_costs_history = []
    best_cost = float('inf')

    base_path = "/home/geriatronics/pmaf_ws/src/planner_optimizer"
    results_path = os.path.join(base_path, "results/svgp")
    figures_path = os.path.join(base_path, "figures/svgp")

    for i in range(num_iterations):
        rec_x = hebo_batch.suggest(n_suggestions=8)
        rospy.loginfo("Iteration {}: Suggested parameters batch:".format(i))
        cost_list = []
        individual_costs_batch = []

        for j in range(len(rec_x)):
            single_x = rec_x.iloc[[j]]
            rospy.loginfo("Processing suggestion {} in batch:".format(j))
            rec_dict = single_x.to_dict(orient='records')[0]
            param_dict = {
                'detect_shell_rad': rec_dict['detect_shell_rad'],
                'agent_mass': 1.0,
                'agent_radius': 0.2,
                'velocity_max': 0.5,
                'approach_dist': 0.25,
                'k_a_ee': [rec_dict[f'k_a_ee_{k}'] for k in range(7)],
                'k_c_ee': [rec_dict[f'k_c_ee_{k}'] for k in range(7)],
                'k_r_ee': [rec_dict[f'k_r_ee_{k}'] for k in range(7)],
                'k_r_force': [0.0]*7,
                'k_d_ee': [rec_dict[f'k_d_ee_{k}'] for k in range(7)],
                'k_manip': [rec_dict[f'k_manip_{k}'] for k in range(7)]
            }
            with open(temp_yaml_file, 'w') as f:
                yaml.dump(param_dict, f)
            total_cost, indiv_costs = run_experiment()
            cost_list.append([total_cost])
            individual_costs_batch.append(indiv_costs)
            if total_cost < best_cost:
                best_cost = total_cost
                best_params = param_dict
            costs.append(total_cost)
            individual_costs_history.append(indiv_costs)
                # Save the best-found parameters and cost to a YAML file
            output_yaml_file = os.path.join(results_path, "best_parameters.yaml")
            best_params['best_cost'] = best_cost
            with open(output_yaml_file, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)
            rospy.loginfo(f"Best parameters and cost saved to {output_yaml_file}")
        cost_array = np.array(cost_list)
        hebo_batch.observe(rec_x, cost_array)
        # Save the global cost evolution plot
        global_cost_plot_path = os.path.join(figures_path, "cost_evolution.png")
        plt.figure(figsize=(8, 6))
        plt.plot(costs, 'x-')
        plt.xlabel("Iterations")
        plt.ylabel("Global Cost")
        plt.title("Global Cost Evolution - GP HEBO")
        plt.savefig(global_cost_plot_path)
        plt.close()
        # Save the individual cost components evolution plot
        individual_costs_plot_path = os.path.join(figures_path, "individual_costs_evolution.png")
        plt.figure(figsize=(8, 6))
        individual_costs_array = np.array(individual_costs_history)
        for idx, label in enumerate(["C_cl", "C_pl", "C_sm", "C_gd"]):
            plt.plot(individual_costs_array[:, idx], label=label)
        plt.xlabel("Iterations")
        plt.ylabel("Individual Costs")
        plt.title("Individual Costs Evolution - GP HEBO")
        plt.legend()
        plt.savefig(individual_costs_plot_path)
        plt.close()

