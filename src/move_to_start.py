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
#q = panda_py.ik(pose)
#print("Initial joint config: ", q)
#position = panda.get_position()

#with open('default_position.yaml', 'w') as f:
#    f.write("ee_position: {}\n".format(position))
 

pose[0,3] += .2
pose[2,3] -= .4
pose[1,3] -= .6
panda.move_to_pose(pose)



#pose = np.array([[0.71634972, -0.69770486,  0.00714188,  0.50698186],
# [-0.69770008, -0.71638117, -0.00355191, -0.60022895],
# [ 0.00759449, -0.00243848, -0.99996819 , 0.19198668],
# [ 0,          0,          0,          1.        ]])


#pose = panda.get_pose()    
#pose[1,3] = -pose[1,3]
panda.move_to_pose(pose)

print("Panda pose: ", pose)

#Move the robot to upright pose
#q = np.zeros_like(q)
#panda.move_to_joint_position(q)
#pose = panda.get_pose()
# Save pose to YAML as rotation matrix and translation vector
#rotation_matrix = pose[:3, :3].tolist()
#translation_vector = pose[:3, 3].tolist()

#with open('EE2base_upright.yaml', 'w') as f:
#    f.write("rotation_matrix:\n")
#    for row in rotation_matrix:
#        f.write("  - {}\n".format(row))
#    f.write("translation_vector: {}\n".format(translation_vector))

#print("Pose saved to EE->base upright.yaml")
