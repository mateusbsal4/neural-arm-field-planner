import numpy as np
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import TransformStamped


def create_tf_matrix_from_msg(transform_msg):
    """
    Create a 4x4 transformation matrix using tf.transformations.
    
    :param transform_msg: TransformStamped message containing translation and rotation.
    :return: 4x4 numpy array representing the transformation matrix.
    """
    # Extract translation
    translation = np.array([
        transform_msg.transform.translation.x,
        transform_msg.transform.translation.y,
        transform_msg.transform.translation.z
    ])
    
    # Extract quaternion
    quaternion = [
        transform_msg.transform.rotation.x,
        transform_msg.transform.rotation.y,
        transform_msg.transform.rotation.z,
        transform_msg.transform.rotation.w
    ]
    
    # Generate the rotation matrix from quaternion
    rotation_matrix = quaternion_matrix(quaternion)  # 4x4 matrix
    # Set the translation
    rotation_matrix[:3, 3] = translation

    return rotation_matrix