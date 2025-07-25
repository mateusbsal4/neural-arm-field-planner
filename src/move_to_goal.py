import panda_py
import numpy as np
import yaml

hostname = '192.168.3.101'
username = 'franka'
password = 'frankaRSI'

desk = panda_py.Desk(hostname, username, password)
desk.unlock()

desk.activate_fci()

print("Franka control interface activated")

panda = panda_py.Panda(hostname)

#Move the robot to its starting pose 
panda.move_to_start()
pose = panda.get_pose()


pose[0,3] += .2
pose[2,3] -= .4
pose[1,3] -= .6
panda.move_to_pose(pose)


pose = panda.get_pose()
pose[1,3] = -pose[1,3]
panda.move_to_pose(pose)

panda.move_to_pose(pose)