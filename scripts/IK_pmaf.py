import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point

class IK_Controller:
    def __init__(self):
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.data_received = False

        # ROS node initializations
        rospy.init_node('ik_genesis_node', anonymous=True)
        self.goal_pos_sub = rospy.Subscriber("agent_position", Point, self.goal_pos_callback)
        self.rate = rospy.Rate(50)  

        # Genesis initialization
        gs.init(backend=gs.gpu)

        # Setup the environment
        self.setup_environment()

        # Build the scene
        self.scene.build()

        # Set control gains
        self.configure_controller()

    def goal_pos_callback(self, data):
        self.data_received = True
        self.target_pos[0] = data.x
        self.target_pos[1] = data.y
        self.target_pos[2] = data.z

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
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.25, 0.3, 0.8), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.25, -0.3, 0.8), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.5, 0.3, 0.6), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.5, -0.3, 0.6), fixed=True))
        self.scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(0.25, -0.1, 0.3), fixed=True))

        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    def configure_controller(self):
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )

    def run(self):
        self.scene.draw_debug_sphere(
            pos=(0.65, 0.0, 0.13),
            radius=0.02,
            color=(1, 0, 0),
        )

        self.end_effector = self.franka.get_link("hand")
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()

        while not rospy.is_shutdown():
            if self.data_received:
                qpos = self.franka.inverse_kinematics(
                    link=self.end_effector,
                    pos=self.target_pos,
                    quat=np.array([0, 1, 0, 0]),
                )
                # Gripper open pos
                qpos[-2:] = 0.04
                self.franka.control_dofs_position(qpos[:-2], np.arange(7))

                # Trajectory visualization
                ee_pos = self.end_effector.get_pos()
                if isinstance(ee_pos, torch.Tensor):
                    ee_pos = ee_pos.cpu().numpy()
                self.scene.draw_debug_line(
                    start=self.prev_eepos,
                    end=ee_pos,
                    color=(0, 1, 0),
                )
                self.prev_eepos = ee_pos

                self.scene.step()

            self.rate.sleep()

if __name__ == "__main__":
    ik_controller = IK_Controller()
    ik_controller.run()