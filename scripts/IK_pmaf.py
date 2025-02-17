import genesis as gs
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Point

target_pos = np.array([0.0, 0.0, 0.0])
data_received = False

def goal_pos_callback(data):
    global target_pos, data_received
    data_received = True
    target_pos[0] = data.x
    target_pos[1] = data.y
    target_pos[2] = data.z



def main():
    # ROS node initializations
    rospy.init_node('ik_genesis_node', anonymous=True)    
    goal_pos_sub = rospy.Subscriber("agent_position", Point, goal_pos_callback)
    rate = rospy.Rate(50)  # 10 Hz
    

    ########################## Genesis init ##########################
    gs.init(backend=gs.gpu)         

    ########################## create a scene ##########################
    scene = gs.Scene(
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

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1, 
            pos=(0.25, 0.3, 0.8),
            fixed = True,
        )
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1, 
            pos=(0.25, -0.3, 0.8),
            fixed = True,
        )
    )   
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1, 
            pos=(0.5, 0.3, 0.6),
            fixed = True,
        )
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1, 
            pos=(0.5, -0.3, 0.6),
            fixed = True,
        )
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05, 
            pos=(0.25, -0.1, 0.3),
            fixed = True,
        )
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    ########################## build ##########################
    scene.build()

    scene.draw_debug_sphere(
        pos=(0.65, 0.0, 0.13),
        radius=0.02,
        color=(1, 0, 0),
    )

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")



    prev_eepos = franka.get_link("hand").get_pos()
    if isinstance(prev_eepos, torch.Tensor):
        prev_eepos = prev_eepos.cpu().numpy()

    while not rospy.is_shutdown():
        if(data_received):
            # move to pre-grasp pose
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=target_pos,
                quat=np.array([0, 1, 0, 0]),
            )
            # gripper open pos
            qpos[-2:] = 0.04
            franka.control_dofs_position(qpos[:-2], motors_dof)

            ##### Trajectory visualization ####
            ee_pos = franka.get_link("hand").get_pos()    
            if isinstance(ee_pos, torch.Tensor):
                ee_pos = ee_pos.cpu().numpy()
            scene.draw_debug_line(
                start=prev_eepos,
                end=ee_pos,
                color=(0, 1, 0),
            )
            prev_eepos = ee_pos
            ##################################

            scene.step()

        rate.sleep()

if __name__ == "__main__":
    main()