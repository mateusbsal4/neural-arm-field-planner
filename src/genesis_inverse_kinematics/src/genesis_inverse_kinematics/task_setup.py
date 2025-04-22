import genesis as gs
import numpy as np
import yaml
import os

def setup_task(randomize=False, config_filename=None):
    base_dir = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/scene/"
    # Automatically generate a unique config filename if none is provided
    if config_filename is None:
        i = 1
        while True:
            generated_filename = f"scene_{i}.yaml"
            full_path = os.path.join(base_dir, generated_filename)
            if not os.path.exists(full_path):  # Check if the file already exists
                config_filename = generated_filename
                break
            i += 1
    config_filename = os.path.join(base_dir, config_filename)
    # Create the scene and camera
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,
        show_FPS=False
    )
    scene.add_entity(gs.morphs.Plane())
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.0, 0.0, 1.0),
        lookat=(0, 0, 0.5),
        fov=45,
        GUI=False
    )
    selected_env = 'office'
    if selected_env == "office":
        # Common parameters
        base_box_height = 0.4
        base_box_width = 0.4
        desk_thickness = 0.05
        wall_height = 2.0
        wall_width = 0.1
        wall_length = 5.0
        r_pillar = 0.15
        box_edge_length = 0.02
        # Default positions
        wall_pos_default = (-0.45, 0.0, wall_height / 2)
        pillar_pos_default = (-0.3, 0.6, wall_height / 2)
        desk_center_default = (0.5, pillar_pos_default[1], base_box_height - desk_thickness / 2)
        desk_size_default = (2 * desk_center_default[0], 2 * (desk_center_default[1] - base_box_width), desk_thickness)
        support1_default = (desk_center_default[0] / 2, desk_center_default[1], (desk_center_default[2] - desk_thickness / 2) / 2)
        support2_default = (1.5 * desk_center_default[0], desk_center_default[1], (desk_center_default[2] - desk_thickness / 2) / 2)
        cube_center_default = (
            box_edge_length / 2,
            desk_center_default[1] + desk_size_default[1] / 2 - box_edge_length,
            desk_center_default[2] + desk_thickness / 2 + box_edge_length / 2
        )
        cracker_box_center_default = (0.05, desk_center_default[1], desk_center_default[2] + desk_thickness / 2)
        goal_pos_default = [cube_center_default[0], cube_center_default[1], cube_center_default[2]]
        if randomize:
            # Clearance margin
            margin = 0.05
            # Randomize wall
            wall_x = np.random.uniform(-0.6, -0.3)
            wall_y = np.random.uniform(-0.2, 0.2)
            wall_pos = (wall_x, wall_y, wall_height / 2)
            # Randomize pillar
            min_pillar_x = wall_x + wall_width/2 + r_pillar + margin
            pillar_x = np.random.uniform(min_pillar_x, min_pillar_x+0.1)     #varies between -0.05 and 0.05
            min_pillar_y = base_box_width/2 + r_pillar + margin
            pillar_y = np.random.uniform(min_pillar_y, min_pillar_y+0.3)     #varies between 0.4 and 0.6
            pillar_pos = (pillar_x, pillar_y, wall_height / 2)
            # Desk dimensions fixed
            half_length = desk_size_default[0] / 2
            half_width  = desk_size_default[1] / 2
            # Randomize desk
            min_desk_center_x = pillar_x + r_pillar + margin + half_length
            desk_y = np.random.uniform(0, 0.4)
            desk_x = (np.random.uniform(min_desk_center_x, min_desk_center_x+0.2) if desk_y >half_width + base_box_width/2
            else np.random.uniform(half_length+base_box_width/2, half_length+base_box_width/2+0.3)) 
            # Desk Z (fixed)
            desk_z = base_box_height - desk_thickness / 2
            desk_center = (desk_x, desk_y, desk_z)
            desk_size   = desk_size_default
            # Support pillars under desk
            support1 = (desk_x - half_width + 0.05, desk_y, (desk_z - desk_thickness / 2) / 2)
            support2 = (desk_x + half_width + 0.05, desk_y, (desk_z - desk_thickness / 2) / 2)
            # Cube on the upper right section of the desk
            cube_x = desk_x - np.random.uniform(0.0, half_length - box_edge_length)
            cube_y = desk_y + np.random.uniform(box_edge_length, half_width - box_edge_length)
            cube_z = desk_z + desk_thickness / 2 + box_edge_length / 2
            cube_center = (cube_x, cube_y, cube_z)
            # Cracker box on the upper left section of the desk
            cracker_box_center_x = desk_x - np.random.uniform(0.0, half_length - box_edge_length)
            cracker_box_center_y =  desk_y - np.random.uniform(box_edge_length, half_width - box_edge_length)
            cracker_box_center = (cracker_box_center_x, cracker_box_center_y, cube_z)
            # Goal position - 1st quadrant, within robot´s reachable WS
            goal_pos = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.5), np.random.uniform(cube_z, cube_z+0.2)]
        else:
            wall_pos    = wall_pos_default
            pillar_pos  = pillar_pos_default
            desk_center = desk_center_default
            desk_size   = desk_size_default
            support1    = support1_default
            support2    = support2_default
            cube_center = cube_center_default
            cracker_box_center = cracker_box_center_default
            goal_pos  = goal_pos_default
        # Build scene
        scene.add_entity(
            gs.morphs.Box(pos=(0, 0.0, 0.2), fixed=True, size=(0.3, base_box_width, base_box_height)),
            surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
        )
        franka = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0.4))
        )
        scene.add_entity(
            gs.morphs.Box(pos=wall_pos, fixed=True, size=(wall_width, wall_length, wall_height)),
            surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
        )
        scene.add_entity(
            gs.morphs.Cylinder(pos=pillar_pos, fixed=True, height=wall_height, radius=r_pillar),
            surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
        )
        scene.add_entity(
            gs.morphs.Box(pos=desk_center, fixed=True, size=desk_size),
            surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
        )
        for supp in [support1, support2]:
            scene.add_entity(
                gs.morphs.Cylinder(pos=supp, fixed=True, height=2 * supp[2], radius=0.05),
                surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
            )
        scene.add_entity(
            gs.morphs.Box(pos=cube_center, size=(box_edge_length, box_edge_length, box_edge_length), fixed=True),
            surface=gs.surfaces.Plastic(color=(0.0, 0.0, 0.0, 1.0))
        )
        scene.add_entity(
            gs.morphs.Mesh(
                file="/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/model/YCB/cracker_box/textured.obj",
                pos=cracker_box_center
            )
        )
        # Save configuration
        config = {
            'wall_pos': wall_pos,
            'pillar_pos': pillar_pos,
            'desk_center': desk_center,
            'desk_size': desk_size,
            'support_pillars': [support1, support2],
            'cube_center': cube_center,
            'cracker_box_center': cracker_box_center,
            'goal_pos': goal_pos
        }
        with open(config_filename, 'w') as f:
            yaml.safe_dump(config, f)
    else:
        pass
    return scene, franka, cam, goal_pos


def recreate_task(config_filename):
    """
    Recreate the office scene using a YAML configuration file.
    """
    base_dir = "/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/scene/"
    config_filename = os.path.join(base_dir, config_filename)
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,
        show_FPS=False
    )
    scene.add_entity(gs.morphs.Plane())
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.0, 0.0, 1.0),
        lookat=(0, 0, 0.5),
        fov=45,
        GUI=False
    )
    # Build office using config
    base_box_height = 0.4
    base_box_width = 0.4
    desk_thickness = 0.05
    wall_height = 2.0
    wall_width = 0.1
    wall_length = 5.0
    r_pillar = 0.15
    box_edge_length = 0.02
    scene.add_entity(
        gs.morphs.Box(pos=(0, 0.0, 0.2), fixed=True, size=(0.3, base_box_width, base_box_height)),
        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0.4))
    )
    scene.add_entity(
        gs.morphs.Box(pos=tuple(config['wall_pos']), fixed=True, size=(wall_width, wall_length, wall_height)),
        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
    )
    scene.add_entity(
        gs.morphs.Cylinder(pos=tuple(config['pillar_pos']), fixed=True, height=wall_height, radius=r_pillar),
        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
    )
    scene.add_entity(
        gs.morphs.Box(pos=tuple(config['desk_center']), fixed=True, size=tuple(config['desk_size'])),
        surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
    )
    for supp in config['support_pillars']:
        scene.add_entity(
            gs.morphs.Cylinder(pos=tuple(supp), fixed=True, height=2 * supp[2], radius=0.05),
            surface=gs.surfaces.Metal(color=(0.5, 0.5, 0.5, 1.0))
        )
    scene.add_entity(
        gs.morphs.Box(pos=tuple(config['cube_center']), size=(box_edge_length, box_edge_length, box_edge_length), fixed=True),
        surface=gs.surfaces.Plastic(color=(0.0, 0.0, 0.0, 1.0))
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file="/home/geriatronics/pmaf_ws/src/genesis_inverse_kinematics/model/YCB/cracker_box/textured.obj",
            pos=config['cracker_box_center']
        )
    )
    goal_pos = config['goal_pos']
    return scene, franka, cam, goal_pos



