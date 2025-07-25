
#!/usr/bin/env python3
import os
import re
import time
import sys
import numpy as np
import rospy
import roslaunch
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray

# Add the directory containing the 'point_net' package to sys.path
# This assumes the script is run from within the ROS package structure
script_dir = os.path.dirname(os.path.abspath(__file__))
package_root_dir = os.path.join(script_dir, '..', 'src') # Adjust based on actual structure
if package_root_dir not in sys.path:
    sys.path.insert(0, package_root_dir)

from point_net.infer_gains import run_inference  
INPUT_DIR = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/inputs"
DEPTH2PTCLOUD_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud.launch'
SIM_STATIC_LAUNCH    = '/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch'
IK_LAUNCH            = '/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/ik_genesis.launch'
PLANNER_LAUNCH       = '/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/launch/main_demo.launch'
TEMP_YAML_FILE       = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
TOL = 0.04

goal_pos = np.zeros((3, 1))
ee_pos   = np.zeros((3, 1))
task_done = False

def cost_callback(msg):
    """
    Callback to handle the received list of costs and signal task completion.
    """
    global task_done
    task_done = True

def goal_pos_callback(msg):
    global goal_pos
    goal_pos[:] = [[msg.x], [msg.y], [msg.z - 0.10365]]

def ee_pos_callback(msg):
    global ee_pos
    ee_pos[:] = [[msg.x], [msg.y], [msg.z]]

def launch_task(scene_name):
    """
    Launch depth->pointcloud, simulation, IK and planner for a given scene,
    wait until `cost_callback` fires, then shut everything down
    and return the Euclidean error between goal and end-effector.
    """
    global task_done
    task_done = False

    ns     = "/"
    uuid_n = roslaunch.rlutil.get_or_generate_uuid(None, False)

    # 1) launch perception
    depth_args = [f"ns:={ns}"]
    depth_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(DEPTH2PTCLOUD_LAUNCH, depth_args)])
    depth_parent.start()
    time.sleep(2)

    # 2) launch static sim
    sim_args = [f"ns:={ns}", "save_cloud:=true", f"scene:={scene_name}"]
    sim_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(SIM_STATIC_LAUNCH, sim_args)])
    sim_parent.start()
    time.sleep(2)

    # 3) launch IK
    ik_args = [f"scene:={scene_name}", "recreate:=false", "evaluate:=false", "bo:=false", "dataset_scene:=false"]
    ik_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(IK_LAUNCH, ik_args)])
    ik_parent.start()

    txt_path = os.path.join(INPUT_DIR, f"{scene_name}.txt")
    while not os.path.exists(txt_path):
        time.sleep(0.1)
    time.sleep(2)  # wait for IK to finish processing

    # run the NN to get gains
    run_inference(scene_name)

    # 5) launch planner
    planner_parent = roslaunch.parent.ROSLaunchParent(uuid_n, [(PLANNER_LAUNCH, [])])
    planner_parent.start()
    rospy.loginfo("Planner node launched.")

    # wait for cost signal
    while not task_done:
        time.sleep(0.1)

    # shut everything down
    depth_parent.shutdown()
    sim_parent.shutdown()
    ik_parent.shutdown()
    planner_parent.shutdown()

    return np.linalg.norm(goal_pos - ee_pos)



if __name__ == "__main__":
    rospy.init_node("full_pipeline", anonymous=True)

    # Subscribers
    rospy.Subscriber("goal_position", Point, goal_pos_callback)
    rospy.Subscriber("tcp_pos",     Point, ee_pos_callback)
    rospy.Subscriber("cost",        Float32MultiArray, cost_callback)

    # Read parameter for first_scene (integer scene ID)
    scene_name = rospy.get_param('~scene_name')
    
    try:
        error = launch_task(scene_name)
        rospy.loginfo(f"Goal pos: {goal_pos.flatten()}")
        rospy.loginfo(f"End-effector pos: {ee_pos.flatten()}")
        rospy.loginfo(f"ERROR: {error:.6f}")
        rospy.loginfo(f"Scene '{scene_name}' → error={error:.4f}")

    except Exception as e:
        rospy.logerr(f"Failed on scene '{scene_name}': {e}")

    
