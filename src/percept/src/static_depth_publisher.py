#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo

def main():
    rospy.init_node('static_depth_publisher')
    
    # Get first messages
    depth_msg = rospy.wait_for_message('camera/depth/image_rect_raw', Image)
    info_msg  = rospy.wait_for_message('camera/depth/camera_info', CameraInfo)

    # Publishers to custom topics
    depth_pub = rospy.Publisher(
        'camera/depth/image_static',
        Image,
        queue_size=1
    )
    info_pub = rospy.Publisher(
        'camera/depth/camera_info_static',
        CameraInfo,
        queue_size=1
    )

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        info_msg.header.stamp  = now
        info_msg.header.frame_id =  "camera_depth_optical_frame"
        depth_pub.publish(depth_msg)
        info_pub.publish(info_msg)
        rate.sleep()

if __name__ == "__main__":
    main()
