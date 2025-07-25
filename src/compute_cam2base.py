#!/usr/bin/env python3

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def load_average_T_cam2tag(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    positions = np.array([
        [d['cam->tag']['position']['x'], 
         d['cam->tag']['position']['y'], 
         d['cam->tag']['position']['z']]
        for d in data
    ])
    avg_pos = np.mean(positions, axis=0)

    quats = np.array([
        [d['cam->tag']['orientation']['x'],
         d['cam->tag']['orientation']['y'],
         d['cam->tag']['orientation']['z'],
         d['cam->tag']['orientation']['w']]
        for d in data
    ])
    avg_quat = R.from_quat(quats).mean().as_quat()
    R_avg = R.from_quat(avg_quat).as_matrix()
    t_avg = avg_pos.reshape(3, 1)

    return np.vstack((np.hstack((R_avg, t_avg)), [[0, 0, 0, 1]]))

def main():
    # File paths
    cam2tag_path = "/home/geriatronics/pmaf_ws/src/apriltag_ros/tf_homogeneous.yaml"


    T_cam2tag = load_average_T_cam2tag(cam2tag_path)
    print(f"T_cam2tag: \n{T_cam2tag}\n")


    # Hardcoded T_tag2EE
    T_tag2base = np.array([[0, 0, 1, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])
    # Compute T_cam2base
    T_cam2base =  T_tag2base @ T_cam2tag

    # Display result
    np.set_printoptions(precision=6, suppress=True)
    print(f"T_cam2base: \n{T_cam2base}\n")

if __name__ == "__main__":
    main()
