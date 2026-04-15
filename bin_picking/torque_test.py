import threading
import rclpy
import time
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rclpy.impl import rcutils_logger
from bin_picking.robot.node import RobotNode
from bin_picking.common.path import get_package_root

def main(args=None):
    # Init
    rclpy.init(args=args)
    robot = RobotNode()
    logger = rcutils_logger.RcutilsLogger('Main')
    root_path = get_package_root()
    plot_path = root_path / 'bin_picking' / 'data' / 'plots'
    logger.info(f'Path{plot_path}')
    os.makedirs(plot_path, exist_ok=True)

    # Create and start the daemon.
    robot_node = threading.Thread(target=robot.spin, daemon=True)
    robot_node.start()

    # Wait until the node is read for work.
    robot.wait_for_service()

    # Get the start pose.
    start_pose = robot.robot_pose

    target = start_pose.copy()
    target['pos'][2] -= 50.0
    target['tvalue'] = 50.0
    target['ref'] = 0

    _ = robot.move(target)

    # Halt the programm until the robot hits the target.
    # Frequency: 200Hz
    logger.info('Waiting for impact.')

    while not robot.readable:
        time.sleep(0.3)
    
    logger.info('Impact')
    states, trqs = robot.sys_state_full
    robot.readable = False

    logger.info('Plot the torque graphs.')
    fig, axes = plt.subplots(7,1, figsize=(10,14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(trqs.shape[0]), trqs[:,i])
        ax.set_ylabel(f'Joint{i+1}')
        ax.axvline(x=np.where(states == 6)[0][0], color='r', linestyle='--')
        ax.axhline(y=np.max(trqs[:,i]), color='g', linestyle='--')
        ax.grid(True)

    axes[-1].set_xlabel('State')

    logger.info('Store plots.')
    plt.tight_layout()
    raw = input('Please enter maxmium force in kg: ')
    cleaned = re.sub(r"[^\d\.\-]", "", raw)
    force = float(cleaned)
    raw = input('Please insert the sensitivity: ')
    cleaned = re.sub(r"[^\d\.\-]", "", raw)
    sensitivity = float(cleaned) 
    plt.savefig(plot_path / f'Joint_torques_with_sensitivity_{sensitivity}_with_force_{force}_kg.png')

    rclpy.shutdown()



if __name__ == '__main__':
    main()
