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
            'pick_test = bin_picking.test.pickup_test:main',
            'offset=bin_picking.test.offset:main',
            'test_jog=bin_picking.robot.node:test',
            "control=bin_picking.test.light_control:main",
            "tune=bin_picking.test.tune_pid:main",
            "show=bin_picking.test.show:main",
            "pick=bin_picking.pipeline.run:main",
            "pick_loop=bin_picking.pipeline.loop:main",
            "hover_bin=bin_picking.test.hover_bin:main",
            "calibrate_bin=bin_picking.test.calibrate_bin:main",
            "show_mesh=bin_picking.test.show_mesh:main"
        ],
    },
)
