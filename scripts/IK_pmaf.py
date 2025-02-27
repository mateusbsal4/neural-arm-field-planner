
import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point
from genesis_inverse_kinematics.evaluate_path import compute_cost
from genesis_inverse_kinematics.task_setup import setup_task

class IK_Controller:
    def __init__(self):
        self.data_received = False
        self.executed_path = []
        self.TCP_path = []
        # ROS node initializations
        rospy.init_node('ik_genesis_node', anonymous=True)
        self.goal_pos_sub = rospy.Subscriber("agent_position", Point, self.goal_pos_callback)
        self.rate = rospy.Rate(10)  

        # Genesis initialization
        gs.init(backend=gs.gpu)

        # Setup the task
        self.scene, self.franka, self.cam, self.target_pos = setup_task()
        self.goal_pos= self.target_pos

        # Build the scene
        self.scene.build()
        self.scene.draw_debug_sphere(
            pos=self.goal_pos,
            radius=0.02,
            color=(1, 1, 0),
        )

        # Set control gains
        self.configure_controller()

    def goal_pos_callback(self, data):
        self.data_received = True
        self.target_pos[0] = data.x
        self.target_pos[1] = data.y
        self.target_pos[2] = data.z

    def configure_controller(self):
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
        )

    def run(self):
        self.end_effector = self.franka.get_link("hand")
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()
        while not rospy.is_shutdown():
            if self.data_received:
                self.scene.draw_debug_sphere(
                    pos=self.target_pos,
                    radius=0.02,
                    color=(1, 0, 0),
                )
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
            #if np.allclose(self.target_pos, self.goal_pos, atol=1e-3):
            #    cost = compute_cost(self.executed_path, self.TCP_path, self.obstacle_centers, self.obs_radius)
            #    print("Path cost: ", cost)
            self.rate.sleep()

if __name__ == "__main__":
    ik_controller = IK_Controller()
    ik_controller.run()