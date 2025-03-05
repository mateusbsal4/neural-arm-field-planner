
import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from genesis_inverse_kinematics.evaluate_path import compute_cost
from genesis_inverse_kinematics.task_setup import setup_task

class IK_Controller:
    def __init__(self):
        self.data_received = False
        self.executed_path = []
        self.TCP_path = []
        # ROS node initializations
        rospy.init_node('ik_genesis_node', anonymous=True)
        self.start_pos_pub = rospy.Publisher("start_position", Point, queue_size=1)
        self.goal_pos_pub = rospy.Publisher("goal_position", Point, queue_size=1)
        self.current_pos_pub = rospy.Publisher("current_position", Point, queue_size=1) 
        self.target_pos_sub = rospy.Subscriber("agent_position", Point, self.target_pos_callback)
        self.pointcloud_pub = rospy.Publisher("/cameras/depth/color/points", PointCloud2, queue_size=1)
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

        # Convert and publish the start position to the PMAF Planner
        self.end_effector = self.franka.get_link("hand")
        start_pos = self.end_effector.get_pos()
        self.start_pos_msg = Point()
        self.start_pos_msg.x = start_pos[0]
        self.start_pos_msg.y = start_pos[1]
        self.start_pos_msg.z = start_pos[2]
        self.start_pos_pub.publish(self.start_pos_msg)

        # Convert and publish the goal position to the PMAF Planner
        self.goal_pos_msg = Point()
        self.goal_pos_msg.x = self.goal_pos[0]
        self.goal_pos_msg.y = self.goal_pos[1]
        self.goal_pos_msg.z = self.goal_pos[2]
        self.goal_pos_pub.publish(self.goal_pos_msg)

        #Render the depth image of the scene and publish it
        _, depth_img, _, _ = self.cam.render(depth=True, segmentation=True, normal=True)
        print("Type of depth image: ", type(depth_img))
        self.pointcloud = self.convert_depth_image_to_pointcloud(depth_img)
        self.pointcloud_pub.publish(self.pointcloud)



        # Set control gains
        self.configure_controller()


    def convert_depth_image_to_pointcloud(self, depth_img):
        # Assuming depth_img is a numpy ndarray with shape (height, width)
        height, width = depth_img.shape
        points = []

        for v in range(height):
            for u in range(width):
                z = depth_img[v, u]
                if z == 0:
                    continue
                x = (u - width / 2) * z / 525.0
                y = (v - height / 2) * z / 525.0
                points.append([x, y, z])

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_frame'

        pointcloud = pc2.create_cloud(header, fields, points)
        return pointcloud


    def target_pos_callback(self, data):
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
        #self.end_effector = self.franka.get_link("hand")
        self.prev_eepos = self.end_effector.get_pos()
        if isinstance(self.prev_eepos, torch.Tensor):
            self.prev_eepos = self.prev_eepos.cpu().numpy()
        while not rospy.is_shutdown():
            self.start_pos_pub.publish(self.start_pos_msg)
            self.goal_pos_pub.publish(self.goal_pos_msg)
            self.pointcloud_pub.publish(self.pointcloud)
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

                current_pos_msg = Point()
                current_pos_msg.x = ee_pos[0]
                current_pos_msg.y = ee_pos[1]
                current_pos_msg.z = ee_pos[2]
                self.current_pos_pub.publish(current_pos_msg)


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