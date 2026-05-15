import rclpy
import threading
import pandas as pd
import numpy as np
import sys
from rclpy.impl import rcutils_logger
from rclpy.client import Client
from pathlib import Path
from typing import Optional, List
from rclpy.node import Node
from bin_picking.common.helper import get_package_root
from kr_msgs.srv import SetDiscreteOutput, MoveLinear, PauseMotion, ResumeMotion, SetCustomFrame, GetRobotPose, GetRobotState, GetSystemFrame, MoveJoint, SetSystemFrame, SelectJoggingFrame
from kr_msgs.msg import SystemState


class RobotNode(Node):

    def __init__(self, name='robot_node'):
        super().__init__(name)
        self.root_path = get_package_root()
        self._lock = threading.Lock()
        self._logger = rcutils_logger.RcutilsLogger('robot')
        self._state_list = []
        self._trq_list = []
        self.readable = False
        # self._clients = {}
        self.setup_node()

    def setup_node(self):
        self._logger.info('Setup clients.')
        self._sdo = self.create_client(SetDiscreteOutput, 'kr/iob/set_digital_output')
        self._mv = self.create_client(MoveLinear, 'kr/motion/move_linear')
        self._pause = self.create_client(PauseMotion, 'kr/motion/pause')
        self._resume = self.create_client(ResumeMotion, 'kr/motion/resume')
        self._grp = self.create_client(GetRobotPose, 'kr/robot/get_robot_pose')
        self._grs = self.create_client(GetRobotState, 'kr/system/get_robot_state')
        self._gsf = self.create_client(GetSystemFrame, 'kr/robot/get_system_frame')
        self._ptp = self.create_client(MoveJoint, 'kr/motion/move_joint')
        self._ssf = self.create_client(SetSystemFrame, 'kr/robot/set_system_frame')
        self._sjf = self.create_client(SelectJoggingFrame, 'kr/motion/select_jogging_frame')
        self._gss = self.create_subscription(SystemState, 'kr/system/state', self._state_cb_fill, 10)

    # Getter
    @property
    def robot_pose(self) -> dict:
        # Create event since this class is running in a daemon
        event = threading.Event()

        # Mutalbe Object
        result = {}

        # Internal callback function
        def cb(future):

            # Get the desired values
            result['pos'] = future.result().pos
            result['rot'] = future.result().rot

            # Trigger the event
            event.set()

        f = self._grp.call_async(GetRobotPose.Request())
        f.add_done_callback(cb)

        # Block the calling thread either until the event was trigger or timeout
        event.wait(3.0)

        return result


    @property
    def sys_state(self):
        pass

    @property
    def sys_state_full(self) -> tuple:
        with self._lock:
            return np.array(self._state_list), np.array(self._trq_list)

    def set_sys_frame(self, pos, rot):
        event = threading.Event()

        def cb(future):
            res = future.result().success
            if res:
                self._logger.info('Set system frame.')
            else: 
                self._logger.info("Couldn't set system frame.")

            event.set()


        msg = SetSystemFrame.Request()
        msg.name = 'tcp'
        msg.ref = 'tfc'
        msg.pos = pos 
        msg.rot = rot

        f = self._ssf.call_async(msg)
        f.add_done_callback(cb)

        if not event.wait(5.0):
            self._logger.info('Timeout.')
        
        
    def select_jf(self, ref: int = 2):
        event = threading.Event()
        msg = SelectJoggingFrame.Request()
        msg.ref = ref

        def cb(future):
            res = future.result().success

            if res:
                self._logger.info('Selected jogging frame.')
            else: 
                self._logger.info("Couldn't set jogging frame.")
            
            event.set()
        
        f = self._sjf.call_async(msg)
        f.add_done_callback(cb)

        
        if not event.wait(5.0):
            self._logger.info('Timeout')

        

    
    def sys_frame(self, target: str, ref: str) -> Optional[np.ndarray]:

        if target == ref:
            self._logger.warning('Can not reference the same frame to it self.')
            return
        
        if not isinstance(target, str):
            self._logger.error(f'Expected target to be type str, got: {type(target)}.')
            return
        
        if not isinstance(ref, str):
            self._logger.error(f'Expected ref type to be str, got: {type(ref)}.')
            return
            
        
        msg = GetSystemFrame.Request()
        msg.name = target
        msg.ref = ref

        val = np.zeros((2,3), dtype=np.float64)
        event = threading.Event()

        def cb(future):
            result = future.result()

            if result.success:
                val[0,:] = result.pos
                val[1,:] = result.rot
            else:
                self._logger.info('Call was unsuccessfull.')
            
            event.set()

        f = self._gsf.call_async(msg)
        f.add_done_callback(cb)

        if not event.wait(5.0):
            self._logger.warning('Timeout.')

        return val


    # Setter

    def move_pnp(self, pos: List[float]):
        result = None
        event = threading.Event()

        msg = MoveJoint.Request()
        msg.jsconf = pos
        msg.ttype = 1
        msg.tvalue = 1.0
        msg.bpoint = 0
        msg.btype = 0
        msg.bvalue = 15.0
        msg.sync = -1.0
        msg.chaining = 1

        def cb(future):
            nonlocal result
            result = future.result().success
            event.set()
        

        f = self._ptp.call_async(msg)
        f.add_done_callback(cb)

        if not event.wait(5.0):
            self._logger.info('Request failed.')
            return result

        self._logger.info('Request successfull.')
        return result



    def move(self, config: dict) -> bool:

        msg = MoveLinear.Request()
        event = threading.Event()
        result = None

        msg.pos = config.get('pos', [0.0,0.0,0.0])
        msg.rot = config.get('rot', [0.0,0.0,0.0])
        msg.ref = config.get('ref', 2)
        msg.ttype = config.get('ttype', 0)
        msg.tvalue = config.get('tvalue', 10.0)
        msg.bpoint = config.get('bpoint', 0)
        msg.btype = config.get('btpye', 0)
        msg.bvalue = config.get('bvalue', 100.0)
        msg.sync = config.get('sync', -1.0)
        msg.chaining = config.get('chain', 1)

        def cb(future):
            nonlocal result 
            result = future.result().success
            event.set()
        
        f = self._mv.call_async(msg)
        f.add_done_callback(cb)

        if not event.wait(3.0):
            self._logger.info('Timeout')
        
        self._logger.info('Message delivered successfully.')

        return bool(result)



    # Internal functions

    # Store the recived message from the system state topic.
    def _state_cb_fill(self, msg):

        with self._lock:
            self._state_list.append(msg.robot_state.val)
            self._trq_list.append(msg.sensed_trq)

            # Set halted, so the external code can check when it robot hits something
            if msg.robot_state.val == 6:
                self.readable = True

    def wait_for_service(self, timeout=5.0):
        self._logger.info('Waiting for servies to startup.')
        for attr in self.__dict__.values():
            if isinstance(attr, Client):
                if not attr.wait_for_service(timeout_sec=timeout):
                    self._logger.error(f'Servce unreachable: {attr.srv_name}.')
                    sys.exit(1)
                else:
                    self._logger.info(f'Service available: {attr.srv_name}.')

    def spin(self):
        self.get_logger().info(f'Node {self.__class__.__name__} start spining.')
        rclpy.spin(self)
    
    def close(self):
        self.destroy_node()
