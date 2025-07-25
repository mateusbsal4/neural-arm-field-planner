#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import yaml
from torch_geometric.data import Data
# compute the absolute path to your src/point_net folder:
here = os.path.dirname(__file__)                # e.g. .../src/point_net
root = os.path.abspath(os.path.join(here, '..'))  # .../src
point_net_src = os.path.join(root, 'point_net')
sys.path.insert(0, point_net_src)
from model import PointNet2

# Hardcoded paths
MODEL_CKPT = "/home/geriatronics/pmaf_ws/src/point_net/checkpoint_epoch_10.pt"
INPUT_DIR = "/home/geriatronics/pmaf_ws/src/dataset_generator/data/inputs"
OUTPUT_YAML = "/home/geriatronics/pmaf_ws/src/multi_agent_vector_fields/config/agent_parameters_temp.yaml"
NPOINTS = 2500

def preprocess_pointcloud(txt_path, npoints=NPOINTS):
    pts = np.loadtxt(txt_path, dtype=np.float32)
    if pts.shape[0] >= npoints:
        choice = np.random.choice(pts.shape[0], npoints, replace=False)
    else:
        choice = np.random.choice(pts.shape[0], npoints, replace=True)
    point_set = pts[choice, :]
    centroid = point_set.mean(axis=0)
    point_set -= centroid
    max_dist = np.max(np.linalg.norm(point_set, axis=1))
    point_set /= max_dist
    pos = torch.from_numpy(point_set).float()
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    return Data(pos=pos, batch=batch)

def run_inference(scene_name, device="cpu"):
    # Model hyperparameters must match training
    model = PointNet2(
        set_abstraction_ratio_1=0.748,
        set_abstraction_ratio_2=0.3316,
        set_abstraction_radius_1=0.4817,
        set_abstraction_radius_2=0.2447,
        dropout=0.1
    )
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device)['model_state_dict'])
    model.eval()
    model.to(device)

    txt_path = os.path.join(INPUT_DIR, f"{scene_name}.txt")
    data = preprocess_pointcloud(txt_path, NPOINTS)
    print("Data object:", data)
    data = data.to(device)
    with torch.no_grad():
        pred = model(data)
    gains = pred.squeeze().cpu().numpy().tolist()  # [36] float

    # Map gains to parameter names for YAML output
    # Adjust indices as per your label order!
    yaml_dict = {
        "detect_shell_rad": gains[0],
        "agent_mass": 1.0,  
        "agent_radius": 0.2, 
        "velocity_max": 0.5, 
        "approach_dist": 0.025,  
        "k_a_ee": gains[1:8],
        "k_c_ee": gains[8:15],
        "k_r_ee": gains[15:22],
        "k_r_force": [0.0]*6,
        "k_d_ee": gains[22:29],
        "k_manip": gains[29:36]
    }

    with open(OUTPUT_YAML, 'w') as f:
        yaml.safe_dump(yaml_dict, f)
    print(f"Saved parameters for scene '{scene_name}' to {OUTPUT_YAML}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_gains.py <scene_name>")
        sys.exit(1)
    scene_name = sys.argv[1]
    print(f"Running inference for scene: {scene_name}")
    run_inference(scene_name)