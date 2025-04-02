#!/usr/bin/env python3

import rospy

import argparse
import utils.troubleshoot as troubleshoot

import tf2_ros
import geometry_msgs.msg

from perception_pipeline import PerceptionPipeline
from perception_node import PerceptionNode

from sensor_msgs.msg import PointCloud2
from utils.camera_helpers import create_tf_matrix_from_msg


class SimPerceptionPipeline(PerceptionPipeline):
    def __init__(self):
        super().__init__()

        # load configs
        self.load_and_setup_pipeline_configs()

        # finish setup
        super().setup()

    def load_and_setup_pipeline_configs(self):
        self.perception_pipeline_config = rospy.get_param("perception_pipeline_config/", None)  
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']


class SimPerceptionNode(PerceptionNode):
    def __init__(self):
        rospy.init_node('sim_perception_node')
        super().__init__()
        
        # Initialize pipeline
        self.pipeline = SimPerceptionPipeline()

        self.setup_ros_subscribers()

    def setup_ros_subscribers(self):
        rospy.loginfo("Setting up subscribers")
        self.subscriber = rospy.Subscriber(
            '/camera/depth/points', PointCloud2, self.static_camera_callback)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        rospy.loginfo("Any errors above?")

    def static_camera_callback(self, msg):
        with self.buffer_lock:
            # Directly store the point cloud message since there's only one camera
            self.pointcloud_buffer = msg

            # Read the camera matrix from the tf buffer
            try:
                transform = self.tfBuffer.lookup_transform("world", "camera_depth_optical_frame", rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logerr("Failed to get transform from world to camera_depth_optical_frame")
            
            # Create the 4x4 transformation matrix 
            tf_matrix = create_tf_matrix_from_msg(transform)
            rospy.loginfo("Callback running")
            print("tf_matrix", tf_matrix)
            # Submit the pipeline task with the single point cloud
            future = self.executor.submit(self.run_pipeline, self.pointcloud_buffer, tf_matrix)

def main():
    node = SimPerceptionNode()

    return node

if __name__ == "__main__":
    try:
        node = main()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()