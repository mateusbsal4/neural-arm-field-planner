import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from sensor_msgs.msg import RegionOfInterest

def publish_depth_image():
    rospy.init_node('depth_image_publisher', anonymous=True)
    # Publisher for the depth image (image_rect topic)
    image_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=1)
    # Publisher for the camera info (camera_info topic)
    camera_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        time_stamp = rospy.Time.now()
        depth_image_msg = create_depth_image_msg(time_stamp)
        camera_info_msg = create_camera_info_msg(time_stamp)
        # Publish the depth image
        image_pub.publish(depth_image_msg)
        # Publish the camera info
        camera_info_pub.publish(camera_info_msg)
        rate.sleep()


def create_depth_image_msg(time_stamp):
    # Load the depth image from the .npy file
    depth_image_path = "/home/geriatronics/pmaf_ws/depth_image.npy"
    depth_img = np.load(depth_image_path)
    print(f"Depth image loaded from {depth_image_path}")
    bridge = CvBridge()
    # Convert the NumPy array to a sensor_msgs/Image
    depth_image_msg = bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
    # Set the timestamp and frame ID
    depth_image_msg.header.stamp = time_stamp
    depth_image_msg.header.frame_id = "camera_depth_optical_frame"
    return depth_image_msg

def create_camera_info_msg(time_stamp):
    cam_info = CameraInfo()
    cam_info.header.stamp = time_stamp
    cam_info.header.frame_id = "camera_depth_optical_frame"
    width = 640
    height = 480
    cam_info.width = width
    cam_info.height = height
    cam_info.distortion_model = "plumb_bob"
    cam_info.D = [0, 0, 0, 0, 0]
    # Compute focal length from the horizontal FOV
    fov_deg = 30.0
    fov_rad = fov_deg * np.pi / 180.0
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx  # assuming square pixels
    cx = width / 2.0
    cy = height / 2.0
    cam_info.K = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
    cam_info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
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

if __name__ == "__main__":
    try:
        publish_depth_image()
    except rospy.ROSInterruptException:
        pass