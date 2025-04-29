#!/usr/bin/env python3
import rospy
import roslaunch
import yaml
import time
import os
import sys
import re

def launch_optimizer(scene):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    # Launch the perception nodes
    depth2ptcloud_launch_file = "/home/geriatronics/pmaf_ws/src/percept/launch/depth_to_ptcloud.launch"
    depth2ptcloud_parent = roslaunch.parent.ROSLaunchParent(uuid, [depth2ptcloud_launch_file])
    depth2ptcloud_parent.start()
    time.sleep(5)  
    sim_static_launch_file = "/home/geriatronics/pmaf_ws/src/percept/launch/sim_static.launch"
    sim_static_parent = roslaunch.parent.ROSLaunchParent(uuid, [sim_static_launch_file])
    sim_static_parent.start()
    time.sleep(5)
    rospy.loginfo("Perception nodes started.")
    opt_launch_file = '/home/geriatronics/pmaf_ws/src/planner_optimizer/launch/bayesian_optimizer.launch'
    opt_args = ['scene:=' + scene, 'include_in_dataset:=true']
    # Launch the optimizer node 
    opt_launcher = roslaunch.parent.ROSLaunchParent(
        uuid,
        [(opt_launch_file, opt_args)]
    )
    opt_launcher.start()
    rospy.loginfo(f"[dataset_generator] bayesian_optimizer for '{scene}' started.")
    opt_launcher.spin()     # spins the optimizer node until its shutdown
    rospy.loginfo(f"[dataset_generator] bayesian_optimizer for '{scene}' has finished.")
    # Stop the perception nodes
    depth2ptcloud_parent.shutdown()
    sim_static_parent.shutdown()
    rospy.loginfo("Perception nodes stopped.")
    return

if __name__ == '__main__':
    rospy.init_node('dataset_generator_node', anonymous=True)
    # Path to your scene YAMLs
    config_dir = '/home/geriatronics/pmaf_ws/src/dataset_generator/data/scene_configs'
    if not os.path.isdir(config_dir):
        rospy.logerr(f"Scene configs folder not found: {config_dir}")
        sys.exit(1)
    labels_csv = '/home/geriatronics/pmaf_ws/src/dataset_generator/data/labels.csv'
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
    if not os.path.isfile(labels_csv):
        open(labels_csv, 'w').close()
        rospy.loginfo(f"Created new labels.csv at {labels_csv}")
    # Iterate in sorted order for reproducibility
    scene_files = sorted(
        (f for f in os.listdir(config_dir) if f.endswith('.yaml')),
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    if not scene_files:
        rospy.logwarn("No scene .yaml files found; nothing to do.")
        sys.exit(0)
    for fname in scene_files:
        scene_name = os.path.splitext(fname)[0]
        rospy.loginfo(f"[dataset_generator] Starting optimizer for scene '{scene_name}'")
        try:
            launch_optimizer(scene_name)
        except Exception as e:
            rospy.logerr(f"Failed on scene '{scene_name}': {e}")
            break
        time.sleep(1)
    rospy.loginfo("[dataset_generator] All scenes processed. Exiting.")