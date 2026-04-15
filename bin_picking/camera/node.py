import subprocess
import os
import time
import rclpy
import signal
import threading
import functools
import numpy as np
from rclpy.impl import rcutils_logger
from rclpy.node import Node
# from rclpy.parameter import parameter_dict_from_yaml_file
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from bin_picking.common.helper import load_dict_from_json, get_package_root
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import RGBD


class Camera(Node):

    def __init__(self, name='rgbd_node', timeout: float=5.0, opt_path: str='', mode: str='default', *args):
        super().__init__(name)

        self._logger = rcutils_logger.RcutilsLogger('rgbd')

        self._cfg_path = get_package_root() / 'bin_picking' / 'data' / 'config'

        self._timeout = timeout
        self._opt_path = opt_path
        self._mode = mode
        self._lock = threading.Lock()

        self._subs = {}

        self.data = {}

        if not self._init_camera():
            self._logger.error("Camera node didn't initialise.")
            raise TimeoutError
        
        # NOTE: The dependency function doen't exist for foxy.
        #if mode != 'default':
            #self._set_params()
        
        self._setup_clients()
        self._logger.info('Ready to start.')
        
    # Close the realsense camera node.
    def close(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    @property
    def available_data(self) -> dict:
        self._require_mode('default', 'rgbd')
        with self._lock:
            return self.data.copy()
    
    def get_camera_info(self, name: str):

        self._require_mode('default')

        matches = sum(['color' in name, 'depth' in name, 'infra' in name])

        if matches != 1:
            raise ValueError(f'No data availale for {name}')
        
        if 'infra' in name:
            temp = [i for i in self.data.keys() if 'infra' in i]
            val = []

            for infra in temp:
                # Skip every infra that isn't camera info
                if not 'camera_info' in infra:
                    continue
                
                with self._lock:
                    val.append(self.data.get(infra, {}))

            return val

        elif 'color' in name:
            with self._lock:
                return self.data.get('color_camera_info', {})
        
        else:
            with self._lock:
                return self.data.get('depth_camera_info', {})

    # Private methods.

    # Start the camera node.
    def _init_camera(self):
        self.process = subprocess.Popen(['ros2', 'run', 'realsense2_camera', 'realsense2_camera_node'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        start = time.time()

        while True:

            if time.time() - start >= self._timeout:
                return False

            res = subprocess.run('ros2 topic list | grep /camera/camera', shell=True, capture_output=True, text=True)

            if res.stdout:
                return True

    def _setup_clients(self):

        self._logger.info(f'Setting up the {self._mode} subscriptions.')

        if self._mode == 'rgbd':
            self._setup_rgbd()
        elif self._mode == 'default':
            self._setup_default()
        else:
            raise RuntimeError(f'Unknown mode.')

    # This methode requires a valid ros yaml file.
    def _set_params(self):
        '''
        This Method requires the yaml file, to be a valid ros yaml file.
        The given file sets up the realsense camera node for its required purpose. 

        Example: 
        /node_name:
            ros__paramters:
                param_name: param_value
        '''

        self._logger.info('Setting up the parameters.')

        # For default.
        if not os.path.exists(self._opt_path):
            raise RuntimeError(f'The parameter file does not exist.')
        
        executer = rclpy.executors.SingleThreadedExecutor()
        executer.add_node(self)

        # Create the param client.
        client = self.create_client(SetParameters, '/camera/camera/set_parameters')
        client.wait_for_service()

        # Fetcht the option provided by the user to set the required paramters.
        req = SetParameters.Request()
        param_dict = parameter_dict_from_yaml_file(self._opt_path, target_node='/camera')
        req.parameters = [val for val in param_dict.values()]
        f = client.call_async(req)
        executer.spin_until_future_complete(f)
        executer.remove_node(self)

        self._logger.info('Successfully set the parameter')

    '''
    This methods sets up the necessary subscriptions, for either the default or rgbd mode. (Maybe later more.)
    Make sure the config file got the right topic name, so it works correctly.

    The config file is expecte to have this interface:
    {
        "sub_name" : "topic_name",
        "sub_name"  : "topic_name",
        ...
    }
    sub_name should include the wanted message type and camera type, everything else can be chosen freely.
    '''

    # Setup the necessary clients for default mode.
    def _setup_default(self):
        # Get the subscription dict
        subs_to_init = load_dict_from_json(self._cfg_path / 'camera_default_subs.json')
        
        # Setup the subscriptions
        for key, val in subs_to_init.items():

            # Check if the value has the right instance.
            if not isinstance(val, str):
                raise ValueError(f'Expected str, but received {type(val)}')

            # Get the corresponding message type and callback function.
            msg_type, cb = self._get_msg_type(val)

            # Create the subscription and add it to the dict.
            self._subs[key] = self.create_subscription(msg_type, val, functools.partial(cb, topic_name=self._get_name(val)), 5)

        self._logger.info('Done with the setup.')

    # NOTE: Add check for the val, if it is a existing topic.
    # Setup the necessary subscriptions for rgbd mode
    def _setup_rgbd(self):
        # Get the subscription dict
        subs_to_init = load_dict_from_json(self._cfg_path / 'camera_rgbd_subs.json')

        # Setup the subscriptions
        for key, val in subs_to_init.items():
            
            # Check if the value has the right instance.
            if not isinstance(val, str):
                raise ValueError(f'Expected str, but received {type(val)}')

            # Get the corresponding message type and callback function.
            msg_type, cb = self._get_msg_type(val)

            # Create the subscription and add it to the dict.
            self._subs[key] = self.create_subscription(msg_type, val, functools.partial(cb, topic_name=self._get_name(val)), 5)
        
        self._logger.info('Done with the subscription setup.')


    # Callback function for every camera info message type.
    def _camera_info_cb(self, msg, topic_name):
        '''
        Message:
            header: Header
            height: uint32
            width: unit32
            distortion_model: str
            d: float64[]
            k: float64[9] shape (3,3)
            p: float64[12] shape (3,4)
            binning_x: unit32
            binning_y: unit32
        '''
        pixel = msg.height, msg.width

        data_dict = {
            'resolution' : pixel,
            'model' : msg.distortion_model,
            'distortion' : np.array(msg.d),
            'intrinsic' : np.array(msg.k),
            'projection' : np.array(msg.p)
        }

        with self._lock:
            self.data[topic_name] = data_dict
        

    # Callback function for every image message type.
    def _image_cb(self, msg, topic_name):
        '''
        Message:
            header: Header
            height: uint32
            width: uint32
            encoding: str
            is_bigendian: uint8
            step: uint32
            data: uint8[]
        '''
        pixel = msg.height, msg.width

        data_dict = {
            'time_stamp': msg.header.stamp,
            'resolution': pixel,
            'encoding' : msg.encoding,
            'is_bigendian' : msg.is_bigendian,
            'step' : msg.step,
            'data' : msg.data
        }

        with self._lock: 
            self.data[topic_name] = data_dict


    # Callback function for evey rgbd message type.
    def _rgbd_cb(self, msg, topic_name):
        '''
        Message:
            header: Header
            rgb_camera_info: CameraInfo
            depth_camera_info: CameraInfo
            rgb: Image
            depth: Image
        '''
        depth_pixel = msg.depth.height, msg.depth.width
        rgb_pixel = msg.rgb.height, msg.rgb.width
        
        if depth_pixel != rgb_pixel:
            raise ValueError(f"RGB resolution and depth aren't aligned.")

        data_dict = {

            'color' : {
                'resolution': rgb_pixel,
                'model' : msg.rgb_camera_info.distortion_model,
                'distortion': msg.rgb_camera_info.d,
                'intrinsic': msg.rgb_camera_info.k,
                'projection': msg.rgb_camera_info.p,
                'encoding': msg.rgb.encoding,
                'is_bigendian': msg.rgb.is_bigendian,
                'step': msg.rgb.step,
                'data': msg.rgb.data
            },

            'depth' : {
                'resolution': depth_pixel,
                'model': msg.depth_camera_info.distortion_model,
                'distortion': msg.depth_camera_info.d,
                'intrinsic': msg.depth_camera_info.k,
                'projection': msg.depth_camera_info.p,
                'encoding': msg.depth.encoding,
                'is_bigendian': msg.depth.is_bigendian,
                'step': msg.depth.step,
                'data': msg.depth.data
            }
        }

        with self._lock:
            self.data[topic_name] = data_dict

    # Enforces that some method can only be called by the right mode, to prevent wrong behavior.
    def _require_mode(self, *states):
        if self._mode not in states:
            raise RuntimeError(f'Called the wrong method for the mode')

    # Matches the name of the node to it's corresponding type and callback function.
    def _get_msg_type(self, name: str) -> tuple:

        if 'camera_info' in name:
            return CameraInfo, self._camera_info_cb
        if 'image' in name:
            return Image, self._image_cb
        if 'rgbd' in name:
            return RGBD, self._rgbd_cb
        
        raise RuntimeError(f'No message type available for {name}')

    # Build the name for the call back function to set the key in the data dict.
    def _get_name(self, name: str) -> str:
        # Only rgbd has camera info and image in the same message.
        if 'rgbd' in name:
            return 'rgbd'

        msg_types = ['camera_info', 'image']
        camera_types = ['infra1', 'infra2', 'color', 'depth']

        # Check every msg type
        for mt in msg_types:
            # If the current message type is in the name, check which camera type it is.If not, then check the next one.

            if mt in name:
                
                for ct in camera_types:
                    # If the current camera type is in the name, return the str. If not, then the camera type is not supported.
                    
                    if ct in name:
                        return ct + '_' + mt
                    else:
                        continue

                # Exhausted every supported camera type.
                raise RuntimeError(f'Unsupported camera type in: {name}.')
            else: 
                continue
        
        # Exhausted every supported message type.
        raise RuntimeError(f'Unsupported message type in: {name}.')
            

    def spin(self):
        self._logger.info(f'Node {self.__class__.__name__} start spining.')
        rclpy.spin(self)