import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'people_avoidance'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'cvxopt'],
    zip_safe=True,
    maintainer='Instructor',
    maintainer_email='instructor@example.com',
    description='Skeleton people-avoidance pipeline for teaching.',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'people_avoidance_node = people_avoidance.people_avoidance_node:main',
            'visualizer = people_avoidance.visualizer2:main',
        ],
    },
)
