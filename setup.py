from setuptools import setup, find_packages

package_name = 'bin_picking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fa',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'torque_test = bin_picking.torque_test:main',
            'test = bin_picking.test:main',
            'camera_test = bin_picking.test.camera_test:main',
            'handeye = bin_picking.common.eye_in_hand_calibration:main',
            'stream = bin_picking.camera.camera:stream',
            'pick_test = bin_picking.test.pickup_test:main'
        ],
    },
)
