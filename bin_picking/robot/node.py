import rclpy
import threading
import numpy as np
import sys
from rclpy.impl import rcutils_logger
from rclpy.client import Client
from typing import Optional, List
from rclpy.node import Node
from bin_picking.common.helper import get_package_root
from kr_msgs.srv import SetDiscreteOutput, MoveLinear, PauseMotion, ResumeMotion, GetRobotPose, GetRobotState, GetSystemFrame, MoveJoint, SetSystemFrame, SelectJoggingFrame, SetAnalogOutput
from kr_msgs.msg import SystemState, JogLinear


class RobotNode(Node):
    """
    ROS2 node that wraps the Kassow robot's service/topic interface.

    All motion and I/O calls are async ROS2 service calls internally, but are
    surfaced here as synchronous-looking methods (using ``threading.Event`` for
    blocking). The node must be spun in a background thread so the ROS2 executor
    can deliver service responses while callers block.

    Usage::

        rclpy.init()
        robot = RobotNode()
        threading.Thread(target=robot.spin, daemon=True).start()
        robot.wait_for_service()
        # ... call move(), move_pnp(), etc.
        rclpy.shutdown()
    """

    def __init__(self, name='robot_node'):
        super().__init__(name)
        self.root_path = get_package_root()
        self._lock = threading.Lock()
        self._logger = rcutils_logger.RcutilsLogger('robot')
        self._state_list = []
        self._trq_list = []
        self._pos = []
        self.readable = False
        # self._clients = {}
        self.setup_node()

    def setup_node(self):
        self._logger.info('Setup clients.')
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
        self._jl = self.create_publisher(JogLinear, 'kr/motion/jog_linear', 10)
        self._sdo = self.create_client(SetDiscreteOutput, "kr/iob/set_digital_output")
        self._sao = self.create_client(SetAnalogOutput, "kr/iob/set_voltage_output")

    # --- Getters ---

    @property
    def robot_pose(self) -> dict:
        """
        Current Cartesian TCP pose from the robot controller.

        Returns:
            dict with keys ``'pos'`` (list of 3 floats, mm) and
            ``'rot'`` (list of 3 floats, XYZ Euler degrees). Returns an empty
            dict on timeout.
        """
        event = threading.Event()
        result = {}

        def cb(future):
            result['pos'] = future.result().pos
            result['rot'] = future.result().rot
            event.set()

        f = self._grp.call_async(GetRobotPose.Request())
        f.add_done_callback(cb)
        event.wait(3.0)

        return result


    @property
    def pose(self):
        """Latest joint-space position received from the SystemState topic (degrees)."""
        with self._lock:
            return self._pos[-1]

    @property
    def sys_state_full(self) -> tuple:
        """
        Full history of robot states and torques as numpy arrays.

        Returns:
            (states, torques) — both (N,) arrays accumulated since node start.
        """
        with self._lock:
            return np.array(self._state_list), np.array(self._trq_list)

    def set_sys_frame(self, pos, rot):
        """
        Define the TCP frame relative to the TFC (tool flange center).

        Args:
            pos: [x, y, z] offset in mm.
            rot: [rx, ry, rz] XYZ Euler angles in degrees.
        """
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

    def resume_motion(self) -> None:
        event = threading.Event()

        def cb(future):
            event.set()

        f = self._resume.call_async(ResumeMotion.Request())
        f.add_done_callback(cb)
        event.wait(3.0)

    def turn_digital_output(self, index: int, value: int) -> None:
        """
        Set a digital output on the robot controller (fire-and-forget).

        Args:
            index: Digital output index (e.g. 6 for the suction valve).
            value: 1 to activate, 0 to deactivate.
        """
        msg = SetDiscreteOutput.Request()

        msg.index = index
        msg.value = value

        def cb(future):
            res = future.result().success

            if res:
                self._logger.info("Output set.")
            else: 
                self._logger.info("Couldn't set output.")
            
        f = self._sdo.call_async(msg)
        f.add_done_callback(cb)

    def apply_voltage(self, index: int, voltage: float) -> None:
        """
        Set an analog output voltage (fire-and-forget).

        Args:
            index:   Analog output channel index.
            voltage: Target voltage in volts.
        """
        msg = SetAnalogOutput.Request()

        msg.index = index
        msg.value = voltage
        def cb(future):
            pass
        
        f = self._sao.call_async(msg)
        f.add_done_callback(cb)

    
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
                self._logger.info('Call was unsuccessful.')
            
            event.set()

        f = self._gsf.call_async(msg)
        f.add_done_callback(cb)

        if not event.wait(5.0):
            self._logger.warning('Timeout.')

        return val


    # Setter

    def move_pnp(self, pos: List[float]) -> bool:
        """
        Joint-space PTP (point-to-point) move to a target configuration.

        Args:
            pos: List of 7 joint angles in degrees (joint0 … joint6).

        Returns:
            True if the move request was accepted by the controller.
        """
        result = None
        event = threading.Event()

        msg = MoveJoint.Request()
        msg.jsconf = pos
        msg.ttype = 0
        msg.tvalue = 100.0
        msg.bpoint = 0
        msg.btype = 0
        msg.bvalue = 100.0
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

        self._logger.info('Request successful.')
        return result



    def move(self, config: dict) -> bool:
        """
        Cartesian linear move to a target pose.

        Args:
            config: dict with optional keys:
                - ``pos``    – [x, y, z] in mm (default [0, 0, 0]).
                - ``rot``    – [rx, ry, rz] XYZ Euler angles in degrees (default [0, 0, 0]).
                - ``ref``    – reference frame index (default 2 = world).
                - ``ttype``  – transition type (default 0).
                - ``tvalue`` – transition value, e.g. speed % (default 250).
                - ``bpoint`` – blending point (default 0).
                - ``bvalue`` – blending value (default 500).
                - ``sync``   – synchronisation value (default -1 = async).
                - ``chain``  – motion chaining flag (default 1).

        Returns:
            True if the move request was accepted by the controller.
        """
        msg = MoveLinear.Request()
        event = threading.Event()
        result = None

        msg.pos = config.get('pos', [0.0,0.0,0.0])
        msg.rot = config.get('rot', [0.0,0.0,0.0])
        msg.ref = config.get('ref', 2)
        msg.ttype = config.get('ttype', 0)
        msg.tvalue = config.get('tvalue', 250.0)
        msg.bpoint = config.get('bpoint', 0)
        msg.btype = config.get('btpye', 0)
        msg.bvalue = config.get('bvalue', 500.0)
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

    def jog_linear(self, vel, rot):
        msg = JogLinear()
        msg.vel = vel
        msg.rot = rot
        self._jl.publish(msg)


    # Internal functions

    # Store the recived message from the system state topic.
    def _state_cb_fill(self, msg):

        with self._lock:
            self._state_list.append(msg.robot_state.val)
            self._trq_list.append(msg.sensed_trq)
            self._pos.append(msg.sensed_pos)

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


def test():
    import time
    rclpy.init()
    robot = RobotNode()
    threading.Thread(target=robot.spin, daemon=True).start()
    start = time.monotonic()
    try:
        while True:
            if (time.monotonic() - start) >= 0.05:
                robot.jog_linear(vel=[0.0,0.0,1.0], rot=[0.0,0.0,0.0])
                start = time.monotonic()
    finally:
        rclpy.shutdown()