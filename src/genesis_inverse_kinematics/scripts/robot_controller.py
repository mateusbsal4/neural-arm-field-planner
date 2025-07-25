#!/usr/bin/env python3
import yaml
from scipy.spatial.transform import Rotation as R
import rospy
import panda_py
from panda_py import controllers
import numpy as np
from numpy.linalg import norm
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from genesis_inverse_kinematics.static_transform_publisher import publish_transforms
import threading


def load_transforms():
    # Load and average cam2tag
    with open('/home/geriatronics/pmaf_ws/src/apriltag_ros/tf_homogeneous.yaml', 'r') as f:
        cam2tag_data = yaml.safe_load(f)
    positions = np.array([
        [d['cam->tag']['position']['x'], 
         d['cam->tag']['position']['y'], 
         d['cam->tag']['position']['z']]
        for d in cam2tag_data
    ])
    avg_pos = np.mean(positions, axis=0)

    quats = np.array([
        [d['cam->tag']['orientation']['x'],
         d['cam->tag']['orientation']['y'],
         d['cam->tag']['orientation']['z'],
         d['cam->tag']['orientation']['w']]
        for d in cam2tag_data
    ])
    avg_quat = R.from_quat(quats).mean().as_quat()
    R_cam2tag = R.from_quat(avg_quat).as_matrix()
    t_cam2tag = avg_pos.reshape(3,1)
    T_cam2tag = np.vstack((np.hstack((R_cam2tag, t_cam2tag)), [0,0,0,1]))
    T_tag2base = np.array([[0, 0, 1, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])
    #T_cam2base:        #this is correct!
    #[[ 0.162134  0.023911 -0.986479  1.59506 ]
    # [ 0.986692 -0.016384  0.161772 -0.079901]
    # [-0.012294 -0.99958  -0.02625   0.302621]
    # [ 0.        0.        0.        1.      ]]
    T_cam2base = T_tag2base @ T_cam2tag
    #print("Tcam2base: ", T_cam2base)
    return T_cam2base


class RobotController:
    def __init__(self):
        rospy.init_node("robot_controller")

        # --- Robot Connection ---
        try:
            hostname = '192.168.3.101'
            desk = panda_py.Desk(hostname, 'franka', 'frankaRSI')
            desk.unlock(); desk.activate_fci()
            self.panda = panda_py.Panda(hostname)
            #self.panda.move_to_start()
            self.start_pos = self.panda.get_position()
        except Exception as e:
            rospy.logerr(f"Failed to connect: {e}")
            raise

        # Goal position definition
        #self.goal_pos = np.array([-0.25, 0.5, 0.25], dtype=np.float64)
        #self.goal_pos = np.array([0.46554439, 0.53537632, 0.16181426])
        self.goal_pos = np.array([0.46554439, 0.53537632, 0.21181426])
        # --- TF Camera Pose ---
        self.cam_pose = load_transforms()
        # --- Data Logging ---
        self.trajectory = []
        self.ee_pos = []
        self.log_path = '/home/geriatronics/pmaf_ws/src/trajectory_log.yaml'
        self.log_path_robot= '/home/geriatronics/pmaf_ws/src/ee_pos_log.yaml'
        rospy.on_shutdown(self.save_trajectory)
        rospy.on_shutdown(self.save_ee_pos)
        # --- Publishers & Subscribers ---con
        self.current_pos_pub = rospy.Publisher("current_position", Point, queue_size=1)
        self.start_pos_pub = rospy.Publisher("start_position", Point, queue_size=10)
        self.goal_pos_pub  = rospy.Publisher("goal_position", Point, queue_size=10)
        self.robot_aabb_pub= rospy.Publisher("robot_aabb", Float32MultiArray, queue_size=10)

        rospy.Subscriber("/agent_position", Point, self.agent_position_callback)

        # --- Shared State ---
        self.target_position = None  # type: Optional[np.ndarray]

        # --- Start Impedance Control Thread ---
        threading.Thread(target=self.impedance_control_loop, daemon=True).start()

        # --- Initial Publications ---
        self.publish_start_pose()
        self.publish_goal_pose()
        self.publish_robot_aabb()

    def agent_position_callback(self, msg: Point):
        """
        Update the desired end-effector position (x, y, z) in base frame.
        """
        self.target_position = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
        self.trajectory.append(self.target_position.tolist())
        ee_position = self.panda.get_position()
        self.ee_pos.append(ee_position.tolist())
        #rospy.loginfo(f"New position target: {self.target_position}")

    def impedance_control_loop(self):
        """
        Persistent Cartesian-impedance control loop at 1 kHz.
        """
        # Wait until panda is readyc
        rospy.sleep(0.5)
        q_current = self.panda.get_orientation()
        # Define impedance matrix: [translational | rotational]
        impedance = np.diag([500.0,500.0,500.0, 1e-2,1e-2,1e-2])
        ctrl = controllers.CartesianImpedance(impedance=impedance,
                                             damping_ratio=1.0,
                                             nullspace_stiffness=1.0,
                                             filter_coeff=1.0)
        self.panda.start_controller(ctrl)

        tol = 3e-2  # 5 mm positional tolerance

        with self.panda.create_context(frequency=1000) as ctx:
            while ctx.ok() and not rospy.is_shutdown():
                
                if self.target_position is None:
                    continue
                x_now = self.panda.get_position()
                error = self.target_position - x_now
                # Send absolute target and current orientation
                ctrl.set_control(self.target_position, q_current)

                # Clear target if reached
                if norm(error) < tol:
                    rospy.loginfo("Position reached, clearing target.")
                    self.target_position = None

        # Stop controller on exit
        self.panda.stop_controller()

    def publish_current_pose(self):
        p = self.panda.get_position()
        msg = Point(*p)
        self.current_pos_pub.publish(msg)

    def publish_start_pose(self):
        self.start_pos_pub.publish(Point(*self.start_pos))

    def publish_goal_pose(self):
        """
        Publish the robot's goal TCP pose.
        """
        # This is a placeholder. Replace with the actual goal pose.
        goal_pos_msg = Point()
        goal_pos_msg.x = self.goal_pos[0]
        goal_pos_msg.y = self.goal_pos[1]
        goal_pos_msg.z = self.goal_pos[2]
        self.goal_pos_pub.publish(goal_pos_msg)

    def publish_robot_aabb(self):
        """
        Publish a hardcoded AABB for the robot.
        """
        aabb = Float32MultiArray()
        #aabb.data = [ -0.15, -0.15, 0.0, 0.15, 0.15, 1.18]
        aabb.data = [0.0, 0.0, 0.0, 0.47, -0.54, 0.5]
        self.robot_aabb_pub.publish(aabb)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            publish_transforms(self.cam_pose)
            self.publish_robot_aabb()
            self.publish_start_pose()
            self.publish_goal_pose()
            self.publish_current_pose()
            rate.sleep()

    def save_trajectory(self):
        """
        Save start, waypoints, and final goal to a YAML file.
        """
        data = {
            'start':    self.start_pos.tolist(),
            'waypoints': self.trajectory,
            'goal':     self.goal_pos.tolist()
        }
        with open(self.log_path, 'w') as f:
            yaml.dump(data, f)
        rospy.loginfo(f"Trajectory saved to {self.log_path}")


    def save_ee_pos(self):
        """
        Save start, ee positions, and final goal to a YAML file.
        """
        print("End pose: ", self.panda.get_pose())
        data = {
            'start':    self.start_pos.tolist(),
            'waypoints': self.ee_pos,
            'goal':     self.goal_pos.tolist()
        }
        with open(self.log_path_robot, 'w') as f:
            yaml.dump(data, f)
        rospy.loginfo(f"Trajectory saved to {self.log_path_robot}")



if __name__ == "__main__":
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

