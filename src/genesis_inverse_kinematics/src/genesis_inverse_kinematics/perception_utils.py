import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
import numpy as np

def create_depth_image_msg(depth_img, timestamp):
    bridge = CvBridge()
    # Convert the depth image to a sensor_msgs/Image
    depth_image_msg = bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
    # Set the timestamp and frame ID
    depth_image_msg.header.stamp = timestamp
    depth_image_msg.header.frame_id = "camera_depth_optical_frame"
    return depth_image_msg


def create_camera_info_msg(timestamp, camera):
    cam_info = CameraInfo()
    cam_info.header.stamp = timestamp
    cam_info.header.frame_id = "camera_depth_optical_frame"
    # Use camera attributes
    width, height = camera.res
    cam_info.width = width
    cam_info.height = height
    cam_info.distortion_model = "plumb_bob"
    cam_info.D = [0, 0, 0, 0, 0]  # Assuming no distortion
    # Intrinsics from the camera
    fx, fy, cx, cy = camera.intrinsics[0][0], camera.intrinsics[1][1], camera.intrinsics[0][2], camera.intrinsics[1][2]
    cam_info.K = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
    # Identity rotation matrix 
    cam_info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
    # Projection matrix
    cam_info.P = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
    cam_info.binning_x = 0
    cam_info.binning_y = 0
    # Region of Interest (full image)
    cam_info.roi.x_offset = 0
    cam_info.roi.y_offset = 0
    cam_info.roi.width = width
    cam_info.roi.height = height
    cam_info.roi.do_rectify = False

    return cam_info


