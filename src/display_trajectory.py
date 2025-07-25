#!/usr/bin/env python3
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def main():
    path = "/home/geriatronics/pmaf_ws/src/trajectory_log_success.yaml"
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    start = np.array(data['start'])
    goal = np.array(data['goal'])
    waypoints = np.array(data['waypoints'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot waypoints
    if waypoints.size > 0:
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'o-', label='Trajectory', color='blue')

    # Plot start and goal
    ax.scatter(*start, c='green', s=80, label='Start', marker='^')
    ax.scatter(*goal, c='red', s=80, label='Goal', marker='x')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Planned Trajectory")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
