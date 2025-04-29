#import genesis as gs
import numpy as np
import yaml
from genesis_inverse_kinematics.task_setup import setup_task


def main():
    num_scenes = 10
    for i in range(num_scenes):
        n_floating_obs = np.random.randint(5, 10)
        setup_task(randomize = True, include_in_dataset=True, n_floating_primitives=n_floating_obs)

if __name__ == "__main__":
    main()  