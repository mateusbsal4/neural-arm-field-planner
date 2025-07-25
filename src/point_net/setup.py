from setuptools import setup, find_packages

setup(
    name='point_net',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'rospy',
        'std_msgs',
    ],
    include_package_data=True,
)
