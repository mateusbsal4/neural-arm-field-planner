#!/usr/bin/env python3
import os
import time
import sys
import rospy
import roslaunch
import numpy as np
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
REALSENSE_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/realsense.launch'
STATIC_DEPTH_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/static_depth.launch'
DEPTH2PTCLOUD_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud_real.launch'
PERCEPTION_PIPELINE_LAUNCH = '/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch'
ROBOT_CONTROLLER_LAUNCH = '/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/launch/robot_controller.launch'
PLANNER_LAUNCH = '/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/launch/main_demo.launch'

goal_pos = np.zeros((3, 1))
ee_pos = np.zeros((3, 1))
task_done = False

def cost_callback(msg):
    global task_done
    task_done = True

def goal_pos_callback(msg):
    global goal_pos
    goal_pos[:] = [[msg.x], [msg.y], [msg.z]]

def ee_pos_callback(msg):
    global ee_pos
    ee_pos[:] = [[msg.x], [msg.y], [msg.z]]

def launch_task(scene_name):
    global task_done
    task_done = False

    ns = "/"
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)


    # 4) Launch Robot Controller
    robot_controller_parent = roslaunch.parent.ROSLaunchParent(uuid, [(ROBOT_CONTROLLER_LAUNCH, [])])
    robot_controller_parent.start()
    time.sleep(2)

    print("Launched controller")

    # 2) Launch depth to pointcloud
    depth_args = [f"ns:={ns}"]
    depth_parent = roslaunch.parent.ROSLaunchParent(uuid, [(DEPTH2PTCLOUD_LAUNCH, depth_args)])
    depth_parent.start()
    time.sleep(2)

    print("Launched depth2ptcloud")

    # 3) Launch Perception Pipeline
    perception_args = [f"ns:={ns}",  "save_cloud:=true", f"scene:={scene_name}"]
    perception_parent = roslaunch.parent.ROSLaunchParent(uuid, [(PERCEPTION_PIPELINE_LAUNCH, perception_args)])
    perception_parent.start()
    time.sleep(2)

    print("Launched percept")


    # 1) Launch Realsense
    realsense_parent = roslaunch.parent.ROSLaunchParent(uuid, [(REALSENSE_LAUNCH, [])])
    realsense_parent.start()
    time.sleep(2)
    print("Launched realsense node")
    # Launch static depth image publisher
    static_depth_pub = roslaunch.parent.ROSLaunchParent(uuid, [(STATIC_DEPTH_LAUNCH, [])])
    static_depth_pub.start()
    time.sleep(2)
    print("Launched static depth image publisher")

    # 5) Run Inference
    txt_path = os.path.join(INPUT_DIR, f"{scene_name}.txt")
    while not os.path.exists(txt_path):
        time.sleep(0.1)
    run_inference(scene_name)

    # 6) Launch Planner
    planner_parent = roslaunch.parent.ROSLaunchParent(uuid, [(PLANNER_LAUNCH, [])])
    planner_parent.start()
    rospy.loginfo("Planner node launched.")

    while not task_done:
        time.sleep(0.1)

    # Shutdown
    realsense_parent.shutdown()
    depth_parent.shutdown()
    perception_parent.shutdown()
    robot_controller_parent.shutdown()
    planner_parent.shutdown()

    return np.linalg.norm(goal_pos - ee_pos)

if __name__ == "__main__":
    rospy.init_node("real_robot_pipeline", anonymous=True)

    rospy.Subscriber("goal_position", Point, goal_pos_callback)
    rospy.Subscriber("current_position", Point, ee_pos_callback)
    rospy.Subscriber("cost", Float32MultiArray, cost_callback)

    scene_name = rospy.get_param('~scene_name', 'real_robot_scene')

    try:
        error = launch_task(scene_name)
        rospy.loginfo(f"Goal pos: {goal_pos.flatten()}")
        rospy.loginfo(f"End-effector pos: {ee_pos.flatten()}")
        rospy.loginfo(f"ERROR: {error:.6f}")
        rospy.loginfo(f"Scene '{scene_name}' → error={error:.4f}")

    except Exception as e:
        rospy.logerr(f"Failed on scene '{scene_name}': {e}")