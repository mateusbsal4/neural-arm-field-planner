from setuptools import setup, find_packages

setup(
    name='genesis_inverse_kinematics',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'rospy',
        'geometry_msgs'
    ],
    include_package_data=True,
)
