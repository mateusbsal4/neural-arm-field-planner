import genesis as gs
import numpy as np
import torch
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(script_dir, "../src/genesis_inverse_kinematics"))  
sys.path.append(parent_dir)  
from evaluate_path import compute_cost

class IK_Controller:
    def __init__(self):
        # Genesis initialization
        gs.init(backend=gs.gpu)

        # Setup the environment
        self.setup_environment()

        # Build the scene
        self.scene.build()

        # Set control gains
        self.configure_controller()

        #Set goal EE position
        self.goal_pos = [0.65, 0.0, 0.13]

        self.executed_path = []
        self.TCP_path = []

    def setup_environment(self):
        # Create a scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            show_viewer=True,
            show_FPS=False,
        )
        self.obs_radius = 0.07
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(gs.morphs.Sphere(radius=self.obs_radius, pos=(0.25, 0.3, 0.8), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=self.obs_radius, pos=(0.25, -0.3, 0.8), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=self.obs_radius, pos=(0.5, 0.3, 0.6), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=self.obs_radius, pos=(0.5, -0.3, 0.6), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=self.obs_radius, pos=(0.25, -0.1, 0.3), fixed=True))

        # Define the obstacle centers
        obstacle_centers_list = [
            np.array([0.25, 0.3, 0.8]),
            np.array([0.25, -0.3, 0.8]),
            np.array([0.5, 0.3, 0.6]),
            np.array([0.5, -0.3, 0.6]),
            np.array([0.25, -0.1, 0.3])
        ]
        self.obstacle_centers = np.array(obstacle_centers_list)


        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    def configure_controller(self):
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )

    def plan_path(self):
        self.scene.draw_debug_sphere(pos=self.goal_pos, radius=0.02, color=(1, 0, 0))

        self.end_effector = self.franka.get_link("hand")
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()

        # Target joint angles 
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array(self.goal_pos),
            quat=np.array([0, 1, 0, 0]),
        )
        # Gripper open
        qpos[-2:] = 0.04

        self.path = self.franka.plan_path(
            qpos_goal=qpos,
            num_waypoints=200, # 2s duration
        )         


    def execute_path(self):
        # Execute the planned path
        for waypoint in self.path:
            self.franka.control_dofs_position(waypoint)

            ee_pos = self.end_effector.get_pos()    # Trajectory visualization
            if isinstance(ee_pos, torch.Tensor):
                ee_pos = ee_pos.cpu().numpy()
            self.TCP_path.append(ee_pos)
            self.scene.draw_debug_line(
                start=self.prev_eepos,
                end=ee_pos,
                color=(0, 1, 0),
            )
            self.prev_eepos = ee_pos

            links_pos = self.franka.get_links_pos()     #Store position of all robot´s links to executed_path
            if isinstance(links_pos, torch.Tensor):
                links_pos = links_pos.cpu().numpy()
            self.executed_path.append(links_pos)
            
            self.scene.step()

        cost = compute_cost(self.executed_path, self.TCP_path, self.obstacle_centers, self.obs_radius)
        print("Path cost: ", cost)

        while True:
            self.scene.step()





if __name__ == "__main__":
    ik_controller = IK_Controller()
    ik_controller.plan_path()
    ik_controller.execute_path()