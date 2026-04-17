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
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from bin_picking.common.helper import load_dict_from_json, get_package_root
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import RGBD #type: ignore
from typing import Tuple


class Camera(Node):
    '''
    Data Structures:

        Image:
            time_step: Time stamp in seconds and nanoseconds
            resolution: The resolution of the corresponding camera.
            encoding: The image encoding.
            is_bigendian: Defines the order of the MSB and LSB.
            step: The index at which the next row starts.
            data: The raw image data, expressed as byte array.
        
        CameraInfo:
            resolution: The resolution of the corresponding camera.
            model: The distortion model.
            distortion: The distortion parameter.
            intrinsic: The intrinsic matrix.
            projection: The projection matrix.
    '''

    def __init__(self, name='rgbd_node', timeout: float=5.0, mode: str='default', *args):
        '''
        Params:
            mode: Available modes are default, or rgbd. Default means color and depth aren't synced and don't share the same coordinate system.
        '''
        super().__init__(name)

        self._logger = rcutils_logger.RcutilsLogger('rgbd')

        self._cfg_path = get_package_root() / 'bin_picking' / 'data' / 'config'
        self._timeout = timeout
        self._mode = mode
        self._lock = threading.Lock()
        self._cfg_file = self._cfg_path / 'camera_parameter_cfg.json'
        self._is_init = False

        if mode == 'default':
            self._condition = [False, False]
        else:
            self._condition = [False, False, False]

        self._subs = {}
        self._data = {}

        self._init_camera()

        # Set params if necessary.
        if mode != 'default':
            self._set_params()
        else:
            self._setup_param_client()
        
        self._setup_subscriptions()
        self._logger.info('Ready to start.')
    
    # Normally you would raise, but this would require to have an custom error, since both of the first if statements would raise the same error.
    def set_video(self, cam: str, width: int, height: int, fps: int, timeout=5.0):
        '''
        Set the video parameter for the given camera.
        The camera needs to be reset after the call. This function does it automatically.
        '''
        result = None
        event = threading.Event()

        def cb(future):

            nonlocal result
            result = future.result().results[0]
            event.set()
        
        def call(req):

            nonlocal result
            event.clear()
            f = self.client.call_async(req)
            f.add_done_callback(cb)

            if not event.wait(timeout=timeout):
                raise TimeoutError(f'Timeout for request: {req}')

            # If code reaches this line, result will always be type SetParametersResult, never None
            if not result.successful: # type: ignore
                return False
            else:
                return True
            
        
        lut = {
            'depth' : 'depth_module.depth_profile',
            'infra': 'depth_module.infra_profile',
            'color': 'rgb_camera.color_profile'
        }

        if cam not in lut:
            self._logger.warning(f'No camera setting available for: {cam}')
            return False

        if self._mode == 'rgbd' and cam != 'infra':
            self._logger.warning('Can not change the setting while in rgbd mode.')
            return False

        cfg = [width, height, fps]
        possible_settings = load_dict_from_json(self._cfg_path / 'video_settings.json').get(cam, [])

        if cfg not in possible_settings:

            self._logger.warning(f'No supported setting for combination: {cfg}')
            return False
        
        # Turn the int arr with the settings into the requires string format
        val = 'x'.join(str(x) for x in cfg)

        # Construct the parameter request.
        resolution = SetParameters.Request()
        resolution.parameters = [Parameter(name=lut[cam], value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=val))]

        # Cameras needs to be reset, based on which resolution should be changed.
        reset_lut = {
            'color': ['enable_color'],
            'infra' : ['enable_infra1', 'enable_infra2'],
            'depth': ['enable_depth']
        }

        cams_to_reset = reset_lut[cam]
        
        # The camera needs to be reset, so the resolution can change to the desired value.
        requests = [resolution] + [SetParameters.Request(
                                                        parameters=[
                                                        Parameter(name=c,
                                                        value= ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=b))])
                                                        for c in cams_to_reset
                                                        for b in [False, True]]
        
        self._logger.info(f'Sending request to change video settings to cam: {cam}, with value: {val}.')
        self._logger.info(f'After the request was successful, the program will automatically reset the affected cameras.')

        for r in requests:
            try:
                suc = call(req=r)

                if not suc:
                    raise RuntimeError
                
            except TimeoutError:
                self._logger.error('Request timed out, please try again later.')
                return False
            
            except RuntimeError:
                self._logger.error(f'Request failed. Reason for the failure: {result.reason}') # type: ignore
                return False
        
        self._logger.info('Successfully changed the resolution.')
        return True



    def close(self):
        '''
        Close the realsense camera node.
        '''
        # Safe guard, if the init raises the timeout, close can't be called.
        if not hasattr(self, 'process'):
            self._logger.info('Can not close something that does not exists.')
            return
        
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    # Wait to prevent empty data.
    def wait_for_init(self, wait: float, max_count: int):
        '''
        Call this method outside the spin thread, else the thread will halt.

        Params:
            wait: Time to wait until the thread continues.
            max_count: Max amount of retries.
        '''

        for attempt in range(max_count):

            self._logger.info(f'Waiting for initialisation, attempt {attempt + 1} / {max_count}.')

            with self._lock:
                
                if all(self._condition):
                    self._logger.info('Data arrived, ready for API calls.')
                    return
                
            time.sleep(wait)
        
        raise TimeoutError(f'Camera did not initialise after {max_count} attemps.')

    @property
    def available_data(self) -> dict:
        with self._lock:
            return self._data.copy()
    
    @property
    def infra_data(self) -> Tuple[dict, bool]:
        '''
        Gets every available infra data.

        Returns:
            dict: The infra data.
            bool: If the operation was successfully.

        Data Structure:
            infraX_camera_info: camera_info
            infraX_image: image
        '''
        with self._lock:
            val = {
                k: self._data.get(k, {}).copy()
                for k in self._data
                if 'infra' in k
            }
            
        return val, bool(val)

    @property
    def color_data(self) -> Tuple[dict, bool]:
        '''
        Gets every available color data.

        Returns:
            dict: The color data.
            bool: If the operation was successfully.

        Data Structure:
            color_camera_info: camera_info
            color_image: image
            
        '''
        with self._lock:

            val = {
                k: self._data.get(k, {}).copy()
                for k in self._data
                if 'color' in k
            }
        
        return val, bool(val)


    @property
    def depth_data(self) -> Tuple[dict, bool]:
        '''
        Gets every available depth data.

        Returns:
            dict: The depth data.
            bool: If the operation was successfully.

        Data Structure:
            depth_camera_info: camera_info
            depth_image: image 
        '''

        with self._lock:
            val = {
                k: self._data.get(k, {}).copy()
                for k in self._data
                if 'depth' in k
            }

        return val, bool(val)

    @property
    def rgbd_data(self) -> Tuple[dict, bool]:
        '''
        Gets the rgbd data.

        Returns:
            dict: The rgbd data.
            bool: If the operation was successfully.

        Data Structure:

            color:
                resolution: height, width
                model: distortion model
                distortion: distortion parameters
                intrinsic: intrinsic matrix, shape(3,3)
                projection: projection matrix, shape(3,4)
                encoding: image encoding
                is_bigendian: defines the order of the MSB and LSB
                step: defines at which index the new row starts
                data: the raw image data as byte array
            depth:
                model: distortion model
                distortion: distortion parameters
                intrinsic: intrinsic matrix, shape(3,3)
                projection: projection matrix, shape(3,4)
                encoding: image encoding
                is_bigendian: defines the order of the MSB and LSB
                step: defines at which index the new row starts
                data: the raw image data as byte array
                
        '''

        try:
            self._require_mode('rgbd')
        except RuntimeError:
            self._logger.error('Wrong mode for this call.')
            return {}, False
        
        with self._lock:
            val = self._data.get('rgbd', {}).copy()

        return val, bool(val)
    

    def get_camera_info(self, name: str) -> Tuple[dict, bool]:
        '''
        Gets the requested camera info.

        Params:
            name: The name of the wanted data, e.g color.
        
        Returns:
            dict: The requested data.
            bool: If the operation was successfully.
        '''
        # Check the request for supported data
        matches = (('color'in name) + ('depth' in name) + ('infra' in name))

        if matches != 1:
            self._logger.warning(f'Unsupported data type: {name}.')
            return {}, False
        
        # Gets every available infra data, if requested.
        if 'infra' in name:

            with self._lock:

                val = {
                    k: self._data.get(k, {}).copy()
                    for k in self._data
                    if 'infra' in k and 'camera_info' in k
                }

            return val, bool(val)

        # Gets either the color or depth camera info
        key = 'color_camera_info' if 'color' in name else 'depth_camera_info'
        with self._lock:
            temp = self._data.get(key, {}).copy()
        return temp, bool(temp)

    def get_image(self, name:str) -> Tuple[dict, bool]:
        '''
        Gets the requested image.

        Params:
            name: The name of the wanted data, e.g depth.
    
        Returns:
            dict: The requested data.
            bool: If the operation was successfully.
        '''

        # Check the request for supported data
        matches = (('color'in name) + ('depth' in name) + ('infra' in name))

        if matches != 1:
            return {}, False
        
        # Gets every available infra image
        if 'infra' in name:

            with self._lock:

                val = {
                    k: self._data.get(k, {}).copy()
                    for k in self._data
                    if 'infra' in k and 'image' in k
                }

            return val, bool(val)

        # Gets either the color image or the depth
        key = 'color_image' if 'color' in name else 'depth_image'
        with self._lock:
            val = self._data.get(key, {}).copy()

        return val, bool(val)


    # Private methods.

    # Start the camera node.
    def _init_camera(self):
        self.process = subprocess.Popen(['ros2', 'run', 'realsense2_camera', 'realsense2_camera_node'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        start = time.time()

        while True:

            if time.time() - start >= self._timeout:
                self._logger.error("Camera node didn't initialise.")
                raise TimeoutError(f'Timeout while initialising the camera.')

            res = subprocess.run('ros2 topic list | grep /camera/camera', shell=True, capture_output=True, text=True)

            # Wait for the camera to setup internally.
            if res.stdout:
                time.sleep(1.0)
                return 
            
            time.sleep(0.2)

    
    def _setup_subscriptions(self):

        self._logger.info(f'Setting up the {self._mode} subscriptions.')

        if self._mode == 'rgbd':
            self._setup_rgbd()
        elif self._mode == 'default':
            self._setup_default()
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    # Get the parameter configuration and set them accordingly.
    def _set_params(self):

        self._logger.info('Setting up the parameters.')

        # For default.
        if not os.path.exists(self._cfg_file):
            raise FileNotFoundError(f'The parameter file does not exist.')
        
        # Spawn an executor for the scope to set the params.
        executor = rclpy.executors.SingleThreadedExecutor() # type: ignore
        executor.add_node(self)

        # Create the param client.
        self.client = self.create_client(SetParameters, 'camera/camera/set_parameters')
        self.client.wait_for_service()

        # Fetch the option provided by the user to set the required paramters.
        req = SetParameters.Request()
        req.parameters = self._get_params_from_file()
        f = self.client.call_async(req)
        executor.spin_until_future_complete(f)
        executor.remove_node(self)

        self._logger.info('Successfully set the parameter')

    
    # This methods sets up the necessary subscriptions, for either the default or rgbd mode. (Maybe later more.)
    # Make sure the config file got the right topic name, so it works correctly.

    # The config file is expected to have this interface:
    # {
    #    "sub_name" : "topic_name",
    #    "sub_name" : "topic_name",
    #    ...
    # }
    # sub_name should include the wanted message type and camera type, everything else can be chosen freely.

    # Setup the necessary clients for default mode.
    def _setup_default(self):
        # Get the subscription dict
        subs_to_init = load_dict_from_json(self._cfg_path / 'camera_default_subs.json')
        
        # Setup the subscriptions
        for key, val in subs_to_init.items():

            # Check if the value has the right instance.
            if not isinstance(val, str):
                raise ValueError(f'Expected str, but received {type(val)}')
            
            if 'rgbd' in val:
                raise RuntimeError(f'Wrong mode for this topic {val}:')

            # Get the corresponding message type and callback function.
            msg_type, cb = self._get_msg_type(val)

            # Create the subscription and add it to the dict.
            self._subs[key] = self.create_subscription(msg_type, val, functools.partial(cb, topic_name=self._get_name(val)), 5)

        self._logger.info('Done with the setup.')

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


    def _setup_param_client(self):

        self.client = self.create_client(SetParameters, 'camera/camera/set_parameters')
        self.client.wait_for_service()

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
            binning_x: uint32
            binning_y: uint32
        '''

        pixel = msg.width, msg.height

        data_dict = {
            'resolution' : pixel,
            'model' : msg.distortion_model,
            'distortion' : np.array(msg.d),
            'intrinsic' : np.array(msg.k),
            'projection' : np.array(msg.p)
        }

        with self._lock:
            self._data[topic_name] = data_dict

            if not self._condition[0]:
                self._condition[0] = True
        

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
            self._data[topic_name] = data_dict

            if not self._condition[1]:
                self._condition[1] = True


    # Callback function for every rgbd message type.
    def _rgbd_cb(self, msg, topic_name):
        '''
        Message:
            header: Header
            rgb_camera_info: CameraInfo
            depth_camera_info: CameraInfo
            rgb: Image
            depth: Image
        '''

        self._require_mode('rgbd')

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
            self._data[topic_name] = data_dict

            if not self._condition[2]:
                self._condition[2] = True

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
        
        raise ValueError(f'No message type available for {name}')

    # Build the name for the call back function to set the key in the data dict.
    def _get_name(self, name: str) -> str:

        # Only rgbd has camera info and image in the same message.
        if 'rgbd' in name:
            return 'rgbd'

        supported_msg_types = ['camera_info', 'image_raw', 'image_rect_raw']
        supported_camera_types = ['infra1', 'infra2', 'color', 'depth']

        # Split the topic name into segements.
        segments = name.split('/')

        camera_type = segments[-2]
        msg_type = segments[-1]

        if camera_type not in supported_camera_types:
            raise ValueError(f'Unsupported camera type in {name}')
        
        if msg_type not in supported_msg_types:
            raise ValueError(f'Unsupported message type in {name}')

        # Map both image types to _image.
        key_msg = 'image' if 'image' in msg_type else msg_type

        return f'{camera_type}_{key_msg}'


    def _get_params_from_file(self):
        '''
        Attention the supported parameter are hard coded, add your desired param here or change the function to be dynamic.
        This method requires the json to be in this format:
        {
            "name_of_param" : bool
        }
        '''

        # Param check is hard coded so it only supports boolean values for the switch to rgbd, given by realsense. If you want to configure other parameter, change the supported params to a function call, which fetches every available parameter.
        # Also add a function, which checks the parameter for it's supposed parametervalue, so you can send int, etc. But don't forget to add the check function for the json.

        # Load the params options from the file.
        opt = load_dict_from_json(self._cfg_path / 'camera_parameter_cfg.json')

        supported_params = ['enable_sync', 'enable_rgbd', 'align_depth.enable', 'enable_color', 'enable_depth']

        params = []

        for key, val in opt.items():

            if not isinstance(val, bool):
                raise ValueError(f'Expected type bool, got: {type(val).__name__} for value.')

            if key not in supported_params:
                raise ValueError(f'Unsupported parameter {key}.')
            
            param_value = ParameterValue()
            param_value.type = ParameterType.PARAMETER_BOOL
            param_value.bool_value = val

            param = Parameter()
            param.name = key
            param.value = param_value

            params.append(param)

        return params

    # Daemon function.
    def spin(self):
        self._logger.info(f'Node {self.__class__.__name__} start spinning.')
        rclpy.spin(self)