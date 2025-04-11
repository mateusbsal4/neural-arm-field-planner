# filepath: /home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/scripts/task_setup.py
import genesis as gs
import numpy as np

def setup_task():
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(3, 0, 1.0), camera_lookat=(0.0, 0.0, 0.5), camera_fov=45, max_FPS=60),
        sim_options=gs.options.SimOptions(dt=0.01), show_viewer=True, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    cam = scene.add_camera(
        res    = (640, 480),
        #pos    = (0.0, 0.0, 0.0),
        pos    = (3.0, 0.0, 1.0),
        lookat = (0, 0, 0.5),
        fov    = 45,
        GUI    = False,
    )
    selected_env = 'office'
    if selected_env == 'factory':   #Simulates a pick and place task between two conveyor belts in a factory environment 
        franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        h_plat = 0.1
        w_plat = 0.3
        l_plat = 2.0
        platform1_center = (0.0, -0.6, 0.3)
        platform2_center = (0.0, 0.6, 0.3)
        platform_size = (l_plat, w_plat, h_plat)
        ## Add conveyor belts to the scene ##
        scene.add_entity(gs.morphs.Box(pos=platform1_center, fixed=True, size=platform_size), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Box(pos=platform2_center, fixed=True, size=platform_size), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        ########################################
        y_left_pillars = platform1_center[1]
        x_left_pillars = (platform1_center[0] - 0.25*l_plat, platform1_center[0], platform1_center[0] + 0.25*l_plat)
        y_right_pillars = platform2_center[1]
        x_right_pillars = (platform2_center[0] - 0.25*l_plat, platform2_center[0], platform2_center[0] + 0.25*l_plat)
        h_pillars = platform1_center[2] - h_plat/2
        ## Add supporting pillars to the scene ##
        scene.add_entity(gs.morphs.Cylinder(pos=(x_left_pillars[0], y_left_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Cylinder(pos=(x_left_pillars[1], y_left_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1),
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Cylinder(pos=(x_left_pillars[2], y_left_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Cylinder(pos=(x_right_pillars[0], y_right_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Cylinder(pos=(x_right_pillars[1], y_right_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        scene.add_entity(gs.morphs.Cylinder(pos=(x_right_pillars[2], y_right_pillars, h_pillars/2), fixed=True, height=h_pillars, radius=0.1), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        ########################################   
        ## Add objects to the left conveyor belt ## 
        r_sphere = 0.01      
        box_edge_length = 0.02
        scene.add_entity(gs.morphs.Sphere(pos=(x_left_pillars[1], y_left_pillars, platform1_center[2]+0.5*h_plat+r_sphere), radius = r_sphere, fixed = True), 
                        surface = gs.surfaces.Plastic(color=(0.0, 0.0, 0.0, 1.0)))
        scene.add_entity(gs.morphs.Box(pos=(x_left_pillars[2], y_left_pillars, platform1_center[2]+0.5*h_plat+box_edge_length), 
                        size = (box_edge_length, box_edge_length, box_edge_length)),
                        surface = gs.surfaces.Plastic(color=(0.0, 0.0, 0.0, 1.0)))
        ## Add containers to the right conveyor belt ##
        box_thickness = 0.01
        box_side = w_plat/2
        box_height = 0.05
        lower_box_center = (platform2_center[0], platform2_center[1], platform2_center[2]+h_plat/2 +box_thickness/2)
        scene.add_entity(gs.morphs.Box(pos=lower_box_center, fixed=True, size=(box_side, box_side, box_thickness)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #lower box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0], lower_box_center[1]-0.5*box_side-0.5*box_thickness, platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_side, box_thickness, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0], lower_box_center[1]+0.5*box_side+0.5*box_thickness, platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_side, box_thickness, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container            
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0]-0.5*box_side-0.5*box_thickness, lower_box_center[1], platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_thickness, box_side, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0]+0.5*box_side+0.5*box_thickness, lower_box_center[1], platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_thickness, box_side, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container     
        lower_box_center = (platform2_center[0]+l_plat/4, platform2_center[1], platform2_center[2]+h_plat/2 +box_thickness/2)
        scene.add_entity(gs.morphs.Box(pos=lower_box_center, fixed=True, 
                        size=(box_side, box_side, box_thickness)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #lower box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0], lower_box_center[1]-0.5*box_side-0.5*box_thickness, platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_side, box_thickness, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0], lower_box_center[1]+0.5*box_side+0.5*box_thickness, platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_side, box_thickness, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container            
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0]-0.5*box_side-0.5*box_thickness, lower_box_center[1], platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_thickness, box_side, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container
        scene.add_entity(gs.morphs.Box(pos=(lower_box_center[0]+0.5*box_side+0.5*box_thickness, lower_box_center[1], platform2_center[2]+h_plat/2+box_height/2), fixed=True, 
                        size=(box_thickness, box_side, box_height)), 
                        surface=gs.surfaces.Plastic(color=(0, 0, 0, 1.0)))  #side box of the container                
        ############################################
    elif selected_env == "office":
        ### Add the the robot mounted on a box ###
        base_box_height = 0.4
        base_box_width = 0.4
        scene.add_entity(gs.morphs.Box(pos=(0, 0.0, 0.2), fixed=True, size=(0.3, base_box_width, base_box_height)),
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))  
        franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos = (0, 0, 0.4)))
        ###########################################
        ### Add box  behind the robot to simulate a wall / window ###
        wall_width = 0.1
        wall_height = 2.0
        wall_length = 5.0
        scene.add_entity(gs.morphs.Box(pos=(-0.45, 0.0, wall_height/2), fixed=True, size=(wall_width, wall_length, wall_height)),
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))  
        ##############################################################
        ##### Add pillar between the window and the desk with objects #####
        pillar_center = (-0.3, 0.6, wall_height/2)
        r_pillar = 0.15
        scene.add_entity(gs.morphs.Cylinder(pos=pillar_center, fixed=True, height=wall_height, radius=r_pillar), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        ###################################################################
        ###### Add desk in front of the pillar ######
        desk_thickness = 0.05
        desk_center = (0.5, pillar_center[1], base_box_height - desk_thickness/2)
        desk_size = (2*desk_center[0], 2*(desk_center[1]-base_box_width),desk_thickness)
        scene.add_entity(gs.morphs.Box(pos=desk_center, fixed=True, size=desk_size),
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))) 
        pillar_center = (desk_center[0]/2, 0.6, (desk_center[2]-desk_thickness/2)/2)
        scene.add_entity(gs.morphs.Cylinder(pos=pillar_center, fixed=True, height=2*pillar_center[2], radius=0.05), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))
        pillar_center = (1.5*desk_center[0], 0.6, (desk_center[2]-desk_thickness/2)/2)
        scene.add_entity(gs.morphs.Cylinder(pos=pillar_center, fixed=True, height=2*pillar_center[2], radius=0.05), 
                        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0)))     
        #############################################
        ###### Add objects on the desk #####
        box_edge_length = 0.02
        cube_center = (box_edge_length/2, desk_center[1]+desk_size[1]/2-box_edge_length, desk_center[2]+desk_thickness/2+box_edge_length/2)
        scene.add_entity(gs.morphs.Box(pos=cube_center, 
                        size = (box_edge_length, box_edge_length, box_edge_length), fixed=True), 
                        surface = gs.surfaces.Plastic(color=(0.0, 0.0, 0.0, 1.0)))
        scene.add_entity(gs.morphs.Mesh(file="/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/model/YCB/cracker_box/textured.obj", pos=(0.05,desk_center[1], desk_center[2]+desk_thickness/2 )))
        
        goal_pos = [cube_center[0], cube_center[1], cube_center[2]]
        #goal_pos = [desk_center[0],desk_center[1], desk_center[2]+desk_thickness/2 + 0.15]
    elif selected_env == 'wall':
        franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        scene.add_entity(gs.morphs.Box(pos=(0.7, 0.0, 0.4), fixed=True, size=(0.9, 0.05, 0.8)))



    #goal_pos = [0.65, 0.0, 0.13]
    return scene, franka, cam, goal_pos