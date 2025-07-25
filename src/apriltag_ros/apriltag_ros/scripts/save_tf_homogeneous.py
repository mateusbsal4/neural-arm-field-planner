#!/usr/bin/env python
import rospy
import tf2_ros
import yaml
import numpy as np
import os

def transform_to_pose_dict(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    return {
        'position': {'x': t.x, 'y': t.y, 'z': t.z},
        'orientation': {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w}
    }

class TfSaver:
    def __init__(self, output_file, cam_frame, tag_frame):
        self.poses = []
        self.output_file = output_file
        self.cam_frame = cam_frame
        self.tag_frame = tag_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Timer(rospy.Duration(0.1), self.save_latest_pose)
        rospy.on_shutdown(self.save_to_yaml)

    def save_latest_pose(self, event):
        try:
            trans = self.tf_buffer.lookup_transform(self.cam_frame, self.tag_frame, rospy.Time(0), rospy.Duration(0.1))
            pose = transform_to_pose_dict(trans)
            self.poses.append({'cam->tag': pose})
        except Exception:
            pass

    def save_to_yaml(self):
        if not self.poses:
            rospy.logwarn("No poses recorded.")
            return
        with open(self.output_file, 'w') as f:
            yaml.dump(self.poses, f, default_flow_style=False)
        rospy.loginfo("Saved {} poses to {}".format(len(self.poses), self.output_file))

if __name__ == '__main__':
    rospy.init_node('save_tf_homogeneous')
    output_file = rospy.get_param('~output_file', os.path.expanduser('~/tf_homogeneous.yaml'))
    cam_frame = rospy.get_param('~cam_frame', 'camera_color_optical_frame')
    tag_frame = rospy.get_param('~tag_frame', 'tag0')
    saver = TfSaver(output_file, cam_frame, tag_frame)
    rospy.spin()
