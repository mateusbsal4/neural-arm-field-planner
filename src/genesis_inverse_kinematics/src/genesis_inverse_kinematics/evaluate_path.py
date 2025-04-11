import genesis as gs
import numpy as np
import torch

def compute_cost(executed_path, TCP_path, obstacle_centers, obs_radius, goal_pos):
    C_cl = 0  # clearance cost
    C_pl = 0  # path length cost 
    C_sm = 0  # smoothness cost 
    C_gd = 0  # goal deviation cost
    collision_penalty = 0  # penalty for collision, needs tuning 
    for i, (link_config, TCP_pos) in enumerate(zip(executed_path, TCP_path)):
        ## Clearance cost ##
        for link_pos in link_config:
            d_link_to_obs = np.linalg.norm(obstacle_centers - link_pos, axis=1) - obs_radius
            min_distance = np.min(d_link_to_obs)
            if min_distance > 0:
               C_cl += 1.0/min_distance
            else: # Collision detected
                C_gd = 10*np.linalg.norm(TCP_pos - goal_pos)
                C_sm *= 0.05
                C_cl *= 0.03
                print("Cost terms ")
                print("Clearance cost ", C_cl/i)
                print("Smoothness cost ", C_sm/i)
                print("Path length cost ", C_pl)
                print("Goal deviation cost ", C_gd)
                J = (C_cl + C_sm)/i + C_pl + C_gd + collision_penalty
                return J
        
        ## Path length cost ##
        if i > 0:
            C_pl += np.linalg.norm(TCP_pos - TCP_path[i - 1])
        ## Smoothness cost ##
        if 0 < i < len(TCP_path) - 1:
            C_sm += np.linalg.norm(TCP_path[i + 1] - 2 * TCP_pos + TCP_path[i - 1]) ** 2
    C_cl /= len(executed_path)  # normalize clearance cost 
    C_sm /= len(executed_path)  #normalize smoothness cost 
    C_sm *= 0.05
    C_cl *= 0.03
    C_gd = 10*np.linalg.norm(TCP_path[-1] - goal_pos)  # reach goal cost
    print("Cost terms ")
    print("Clearance cost ", C_cl)
    print("Smoothness cost ", C_sm)
    print("Path length cost ", C_pl)
    print("Goal deviation cost ", C_gd)
    J = C_cl + C_pl + C_sm + C_gd # combine costs
    return J