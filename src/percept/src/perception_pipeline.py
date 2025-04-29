import cupoch as cph
import numpy as np
import cupy as cp

import rospy
import time
import utils.troubleshoot as troubleshoot

import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from std_msgs.msg import Float64MultiArray

class PerceptionPipeline():
    def __init__(self):
        self.check_cuda()
        # Create process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        # Create thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def check_cuda(self):
        """Check if CUDA is available using nvidia-smi"""
        try:
            output = subprocess.check_output(["nvidia-smi"])
            rospy.loginfo("CUDA is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            rospy.logerr("CUDA is not available - nvidia-smi command failed")
            raise RuntimeError("CUDA is required for this pipeline")

    def setup(self):
        # set scene props
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.scene_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # set voxel props
        cubic_size = self.cubic_size
        voxel_resolution = self.voxel_resolution
        self.voxel_size = cubic_size/voxel_resolution
        self.voxel_min_bound = (-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0)
        self.voxel_max_bound = (cubic_size/2.0, cubic_size/2.0, cubic_size/2.0)

    def parse_pointcloud(self, pointcloud_msg, tf_matrix=None, downsample: bool = False, log_performance: bool = False):
        """
        Parse a single point cloud message and process it.
        """
        start = time.time()
        try:
            # Load point cloud from ROS message
            pcd = cph.geometry.PointCloud()
            temp = cph.io.create_from_pointcloud2_msg(
                pointcloud_msg.data,
                cph.io.PointCloud2MsgInfo.default_dense(
                    pointcloud_msg.width, pointcloud_msg.height, pointcloud_msg.point_step
                )
            )
            pcd.points = temp.points
    
            # If a transformation matrix is provided, transform to world-frame
            if tf_matrix is not None:
                pcd = pcd.transform(tf_matrix)

            # Crop point cloud according to scene bounds
            pcd = pcd.crop(self.scene_bbox)

            # Optionally downsample the point cloud
            if downsample:
                every_n_points = 3
                pcd = pcd.uniform_down_sample(every_n_points)
    
            if log_performance:
                rospy.loginfo(f"PointCloud Parsing (CPU+GPU) [sec]: {time.time() - start}")
    
            return pcd
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))
            return None

    def perform_robot_body_subtraction(self, pointcloud, log_performance: bool = False):
        start = time.time()
        filtered_points = pointcloud
        try:
            # Use the published AABB
            aabb = self.robot_aabb
            #rospy.loginfo("Using robot AABB: min {} | max {}".format(aabb.get_min_bound(), aabb.get_max_bound()))
            # Get indices of points within the AABB
            indices = aabb.get_point_indices_within_bounding_box(filtered_points.points)
            # Remove these points from the point cloud
            filtered_points = filtered_points.select_by_index(indices, invert=True)
            # Log how many points were removed
            #if len(filtered_points.points) < len(pointcloud.points):
            #    removed = len(pointcloud.points) - len(filtered_points.points)
            #    rospy.loginfo(f"Robot body subtraction removed {removed} points.")
            #else:
            #    rospy.logwarn("Robot body subtraction had no effect; no points were removed. " +
            #                  "Check coordinate frames and AABB values.")
        except Exception as e:
            rospy.logerr("Failed to process robot body subtraction: {}".format(
                troubleshoot.get_error_text(e, print_stack_trace=False)
            ))
        if log_performance:
            rospy.loginfo(f"Robot body subtraction (GPU) [sec]: {time.time() - start}")
        return filtered_points


    def perform_voxelization(self, pcd:cph.geometry.PointCloud, log_performance:bool=False):
        start = time.time()
        voxel_grid = cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=self.voxel_size,
            min_bound=self.voxel_min_bound,
            max_bound=self.voxel_max_bound,
        )
        if log_performance:
            rospy.loginfo(f"Voxelization (GPU) [sec]: {time.time()-start}")

        return voxel_grid

    
    def convert_voxels_to_primitives(self, voxel_grid:cph.geometry.VoxelGrid, log_performance:bool=False):
        start = time.time()
        voxels = voxel_grid.voxels.cpu()
        primitives_pos = np.array(list(voxels.keys()))

        if primitives_pos.size == 0:  # Handle empty voxel grid
            rospy.logwarn("No voxels found in voxel grid")
            return None
        
        # Transfer data to GPU
        primitives_pos_gpu = cp.asarray(primitives_pos)
        offset = cp.asarray(voxel_grid.get_min_bound())
        voxel_size = cp.asarray(self.voxel_size)
        
        # Compute minimums for each column on GPU
        mins = cp.min(primitives_pos_gpu, axis=0)
        
        # Perform operations on GPU
        primitives_pos_gpu = primitives_pos_gpu - mins[None, :]
        primitives_pos_gpu = primitives_pos_gpu * voxel_size
        primitives_pos_gpu = primitives_pos_gpu + (offset + voxel_size/2)
        
        # Transfer result back to CPU
        primitives_pos = cp.asnumpy(primitives_pos_gpu)

        if log_performance:
            rospy.loginfo(f"Voxel2Primitives (CPU+GPU) [sec]: {time.time()-start}")
        return primitives_pos


    def publish_primitives(self, primitives_pos):
        sphere_msg = Float64MultiArray()
        flat_data = []
        for pos in primitives_pos:
            flat_data.extend([pos[0], pos[1], pos[2], self.voxel_size])
        sphere_msg.data = flat_data
        #print("Publishing voxel grid with size:", len(flat_data))
        self.voxel_grid_pub.publish(sphere_msg)


    def run_pipeline(self, pointcloud_msg, tf_matrix, log_performance: bool = False):
        """
        Run the perception pipeline for a single point cloud.
        """
        start = time.time()
        # Parse the single point cloud
        pointcloud = self.parse_pointcloud(pointcloud_msg, tf_matrix, downsample=True, log_performance=log_performance)
        # Subtract robot body 
        pointcloud = self.perform_robot_body_subtraction(pointcloud, log_performance=log_performance)
        # Perform voxelization
        voxel_grid = self.perform_voxelization(pointcloud, log_performance=log_performance)
        # Convert voxels to primitives
        primitives_pos = self.convert_voxels_to_primitives(voxel_grid, log_performance=log_performance)
        #print("Type of primitives_pos:", type(primitives_pos))
        #print("Shape of primitives_pos:", primitives_pos.shape)
        # Publish primitives
        self.publish_primitives(primitives_pos)


        log_performance = False
        if log_performance:
            rospy.loginfo(f"Perception Pipeline (CPU+GPU) [sec]: {time.time() - start}")
        return primitives_pos
