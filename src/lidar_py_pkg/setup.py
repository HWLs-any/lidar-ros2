from setuptools import find_packages, setup

package_name = 'lidar_py_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a-s',
    maintainer_email='a-s@gmail.com',
    description='LiDAR processing nodes (fusion, filtering, detection) for ROS 2',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'cloud_monitor = lidar_py_pkg.cloud_monitor:main',
            'cloud_filter = lidar_py_pkg.cloud_passthrough_filter:main',
            'cloud_fusion = lidar_py_pkg.cloud_fusion:main',
            'cloud_detector = lidar_py_pkg.cloud_detector:main',
            'plane_remove_o3d = lidar_py_pkg.plane_remove_o3d:main',
            'metrics_csv_logger = lidar_py_pkg.metrics_csv_logger:main',
        ],
    },
)
