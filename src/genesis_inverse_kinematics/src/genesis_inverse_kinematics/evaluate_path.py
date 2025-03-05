import genesis as gs
import numpy as np
import torch

def compute_cost(executed_path, TCP_path, obstacle_centers, obs_radius):
        C_cl = 0  # clearance cost
        C_pl = 0  # path length cost 
        C_sm = 0  # smoothness cost 

        for i, (link_config, TCP_pos) in enumerate(zip(executed_path, TCP_path)):
            ## Clearance cost ##
            for link_pos in link_config:
                d_link_to_obs = np.linalg.norm(obstacle_centers - link_pos, axis=1) - obs_radius
                min_distance = np.min(d_link_to_obs)
                inverse_min_distance = 1.0 / min_distance if min_distance != 0 else float('inf')
                C_cl += inverse_min_distance

            ## Path length cost ##
            if i > 0:
                C_pl += np.linalg.norm(TCP_pos - TCP_path[i - 1])

            ## Smoothness cost ##
            if 0 < i < len(TCP_path) - 1:
                C_sm += np.linalg.norm(TCP_path[i + 1] - 2 * TCP_pos + TCP_path[i - 1]) ** 2

        C_cl /= len(executed_path)  # normalize clearance cost 
        C_sm /= len(executed_path)  #normalize smoothness cost 

        J = C_cl + C_pl + C_sm  # combine costs
        return J