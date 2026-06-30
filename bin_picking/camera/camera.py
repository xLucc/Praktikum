import time
import sys
import logging
import signal
import itertools
import pyrealsense2 as rs
import numpy as np
from multiprocessing import Process, Lock, Value
from multiprocessing.shared_memory import SharedMemory
import cv2 as cv
from typing import Optional
from bin_picking.camera.camera_interface import Camera
from bin_picking.common.helper import load_dict_from_json, get_project_dir


class RealSenseCamera(Camera):
    """
    Thin wrapper around pyrealsense2 for the bin-picking pipeline.

    Supports two acquisition modes:
    - **Direct** (default): `get_depth()` / `get_color()` block and return frames on demand.
    - **Buffered** (`buffering=True`): a background process continuously captures frames
      into shared memory; callers read the latest via `newest_frame`.

    Args:
        serial:    RealSense serial number. If empty the first available device is used.
        adv:       Filename of a RealSense advanced-settings JSON preset (e.g. ``high_density.json``),
                   resolved relative to ``data/realsense_config/``. D400 series only.
        align:     If True, depth frames are aligned to the color frame before returning.
        cfg:       Filename of a stream-configuration JSON in ``data/realsense_config/``.
                   Falls back to ``setup_cfg.json``, then built-in defaults (640×480 @ 30 fps).
        buffering: Enable shared-memory buffering mode.
    """

    # To get unique names for the logger.
    _counter = itertools.count()
    _DEFAULT_STREAM_CONFIG = {"color": {"resolution" : [640, 480], "fps" : 30}, "depth": {"resolution": [640, 480], "fps": 30}}

    def __init__(self, serial: Optional[str]='', adv: Optional[str]='', align: Optional[bool]=True, cfg: Optional[str]=None, buffering: bool = False):
        
        self._ctx = rs.context()
        self._dual = align
        
        # Logger will get the name RealSenseCamera_numberOfObject. Starts with 0.
        self.logger = logging.getLogger(f'{__class__.__name__}_{self._counter}')

        if align:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        if serial:
            self._serial = serial
        else:
            self._get_serial()


        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._product_line = self._get_line()
        self._config.enable_device(self._serial)

        self._root = get_project_dir()
        self._data_dir = self._root / 'data' 
        self._config_path = self._data_dir / 'realsense_config'
        self._config_path.mkdir(parents=True, exist_ok=True)
        self._setup_cfg = self._config_path / 'setup_cfg.json'

        if cfg is not None:
            cfg_path = self._config_path / cfg
            if not cfg_path.exists():
                self.logger.warning("File not found. Using default.")
                self._sensors_to_setup = self._DEFAULT_STREAM_CONFIG.copy()
            self._sensors_to_setup = load_dict_from_json(cfg_path)
        elif self._setup_cfg.exists():
            self._sensors_to_setup = load_dict_from_json(self._setup_cfg)
        else:
            self.logger.info('No config found, using defaults.')
            self._sensors_to_setup = self._DEFAULT_STREAM_CONFIG.copy()
        

        if not self._setup_cfg.exists():
            raise FileNotFoundError('No setup file found.')

        if adv:
            if not adv.endswith('.json'):
                raise RuntimeError(f'Wrong config file type. Needs to be json')
            
            self._adv_settings = self._config_path / adv
            self.logger.info('Found advanced settings.')
            self._set_adv()

        self._setup_sensors()
        
        if buffering:
            self._buf = True
            self._init_buf()
        else:
            self._buf = False
            self._profile = self._pipeline.start(self._config)  
            self._warm_up()

        self._unit = self._dev.first_depth_sensor().get_depth_scale()
        self.logger.info('Done with initialisation.')

    def __del__(self):
        self.logger.info('Clean up.')

        if hasattr(self, '_profile'):
            self._pipeline.stop()
    

    def start_stream(self):
        time.sleep(0.01)
        self._p.start()

    def stop_stream(self):
        if self._p.is_alive():
            self._p.terminate()

    def stop(self):
        """Stop the RealSense pipeline. Only valid in direct (non-buffered) mode."""
        self._mode_guard()
        self.logger.info('Stop the pipeline.')
        self._pipeline.stop()

    def start(self):
        """Start the RealSense pipeline. Only valid in direct (non-buffered) mode."""
        self._mode_guard()
        self.logger.info('Start pipeline.')
        self._profile = self._pipeline.start(self._config)

    def get_depth(self, num_frames: int = 1) -> list:
        """
        Capture ``num_frames`` raw uint16 depth frames (aligned to color if ``align=True``).

        Returns a list of H×W uint16 numpy arrays in depth-sensor units (multiply by
        ``self.unit`` to get metres).
        """
        self._mode_guard()
        return self._get_data(num_frames=num_frames, mode='depth')

    def get_color(self, num_frames: int = 1) -> list:
        """
        Capture ``num_frames`` BGR uint8 color frames.

        Returns a list of H×W×3 uint8 numpy arrays.
        """
        self._mode_guard()
        return self._get_data(num_frames=num_frames, mode='color')
    
    @property
    def color_resolution(self) -> tuple:
        return self._sensors_to_setup['color']['resolution']
    
    @property
    def unit(self):
        return self._unit

    @property
    def newest_frame(self):
        with self._lock:
            depth = self._depth[self._idx.value % 3 - 1, :].copy()
            color = self._color[self._idx.value % 3 - 1, :].copy()
        
        return color, depth
    
    @property
    def shm(self):
        return self.depth_shm.name , self.color_shm.name
    
    @property
    def index(self):
        return self._idx
    

    def stream(self) ->np.ndarray:

        '''
        Streams the color image.
        
        Returns:
            img (np.ndarray): If wanted return the last img.

        Raises:
            KeyboardInterrupt: To exit the stream.
        '''

        self._mode_guard()

        self.logger.info('To save the image, press s. \n To quit the stream, press q.')

        while True:

            frame = self._pipeline.wait_for_frames().get_color_frame()

            if not frame:
                continue
            

            img = np.asanyarray(frame.get_data()).astype(np.uint8)
            cv.imshow('img', img)
            key = cv.waitKey(1) & 0xFF

            if key == ord("s"):
                cv.destroyAllWindows()
                return img
            elif key == ord("q"):
                cv.destroyAllWindows()
                raise KeyboardInterrupt


    def stream_parallel(self, name, lock, frame_event, shape, exit_event):
        self._mode_guard()
        self.start()
        shm = SharedMemory(name=name, create=False)
        img = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


        def handle_sigterm(signum, frame):
            exit_event.set()

        # Catch the sigterm and close the shared memory.
        signal.signal(signal.SIGTERM, handle_sigterm)

        while not exit_event.is_set():
            frame = self._pipeline.wait_for_frames().get_color_frame()

            if not frame:
                continue
            
            acquired = lock.acquire(timeout=1.0)

            if not acquired:
                continue
            
            try:
                img[:] = np.asanyarray(frame.get_data())
            finally:
                lock.release()
            
            frame_event.set()

        shm.close()
        self.stop()


    def _stream_buf(self, h, w):

        stop = False

        def handle_sigterm(signum, frame):
            nonlocal stop
            stop = True

        self._profile = self._pipeline.start(self._config)
        self._warm_up()

        depth_shm = SharedMemory(name=self.depth_shm.name, create=False)
        color_shm = SharedMemory(name=self.color_shm.name, create=False)
        
        depth = np.ndarray(shape=(3, w, h), dtype=np.uint16, buffer=depth_shm.buf)
        color = np.ndarray(shape=(3, w, h, 3), dtype=np.uint8, buffer=self.color_shm.buf)

        last = time.perf_counter()
        signal.signal(signal.SIGTERM, handle_sigterm)

        while not stop:

            if (time.perf_counter() - last) < 0.01:
                continue

            try:
                frame = self._pipeline.wait_for_frames()
            except RuntimeError:
                logging.getLogger(__name__).error("Frames didn't arrive.")
                depth_shm.close()
                color_shm.close()
                sys.exit(1)

            if self._dual:
                frame = self._align.process(frame)
            
            with self._lock:
                depth[self._idx.value % 3, :] = np.asanyarray(frame.get_depth_frame().get_data())
                color[self._idx.value % 3, :] = np.asanyarray(frame.get_color_frame().get_data())

                self._idx.value += 1
            
            last = time.perf_counter()

        depth_shm.close()
        color_shm.close()
        


    def _get_data(self, num_frames, mode):
        frames = []
        self._fill_frames(num_frames, frames)

        if self._dual:
            frames = [self._align.process(f) for f in frames]

        if mode == 'depth':
            return [np.asarray(d.get_depth_frame().get_data()) for d in frames]
        else:
            return [np.asarray(c.get_color_frame().get_data()) for c in frames]




    def _fill_frames(self, num_frames, frames):

        for _ in range(num_frames):
            frame = self._pipeline.wait_for_frames()

            if not (frame.get_depth_frame() and frame.get_color_frame()):
                continue

            frames.append(frame)
            time.sleep(0.015)


    def _init_buf(self):
        h, w = max(self._sensors_to_setup['color']["resolution"], self._sensors_to_setup["depth"]["resolution"])
        size = h * w * 3
        self.depth_shm = SharedMemory(create=True, size=size * np.dtype(np.uint16).itemsize)
        self.color_shm = SharedMemory(create=True, size=size * np.dtype(np.uint8).itemsize * 3)

        self._depth = np.ndarray((3, w, h), dtype=np.uint16, buffer=self.depth_shm.buf)
        self._color = np.ndarray((3, w, h, 3), dtype=np.uint8, buffer=self.color_shm.buf)

        self._p = Process(target=self._stream_buf, args=(h, w, ), daemon=True)
        self._idx = Value("i", 0)
        self._lock = Lock()



    def _get_serial(self):
        devices = self._ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError('No camera connected.')
        
        # Need to raise to verify if the device is busy.
        # Intel didn't implement a function to check that directly!
        if len(devices) > 1:
            for dev in devices:
                serial = dev.get_info(rs.camera_info.serial_number)
                cfg = rs.config()
                cfg.enable_device(serial)
                p = rs.pipeline()

                try:
                    p.start(cfg)

                except RuntimeError as e:
                    if 'busy' in str(e).lower():
                        self.logger.warning(f'Device {serial} is busy.')
                        continue
                    raise e
                
                finally:
                    try:
                        p.stop()
                    except Exception:
                        pass
                
                self._serial = serial
                break
        else:
            # Sets if only 1 device is available.
            self._serial =  devices[0].get_info(rs.camera_info.serial_number)

    def _set_adv(self):

        # Only available for D400 Series.
        if not 'D400' in self._product_line:
            self.logger.warning('No advanced settings available for this device.')
            return
        
        self._advanced_mode = rs.rs400_advanced_mode(self._dev)

        # Start the advanced mode.
        while not self._advanced_mode.is_enabled():
            self._advanced_mode.toggle_advanced_mode(True)
            time.sleep(3.0)
            self._advanced_mode = rs.rs400_advanced_mode(self._dev)

        # Set the advanced settings.
        with open(self._adv_settings, 'r') as f:
            json_str = f.read().strip()
        self._advanced_mode.load_json(json_str)
        self.logger.info('Successfully set the advanced settings.')


    def _get_line(self):
        # Find the corresponding device to the serial.
        try:
            dev = next(d for d in self._ctx.devices if d.get_info(rs.camera_info.serial_number) == self._serial)
        except StopIteration:
            self.logger.error('The given Serial is wrong.')
            raise
        
        self._dev = dev
        return dev.get_info(rs.camera_info.product_line)

    def _setup_sensors(self):

        if self._align:
            if not (self._sensors_to_setup['color']['resolution'] == self._sensors_to_setup['depth']['resolution'] and self._sensors_to_setup['color']['fps'] == self._sensors_to_setup['depth']['fps']):
                self._get_user_input()


        values = ['fps', 'resolution']
        
        for k, v in self._sensors_to_setup.items():
            missing = set(values) - set(v.keys())

            if len(missing) != 0:
                raise ValueError(f'Missing values.')

            if not isinstance(v['resolution'], list):
                raise TypeError(f"Expected resolution to be tuple, got: {type(v['resolution'])}")
            
            if len(v['resolution']) > 2:
                raise ValueError(f"Resolution can only contain 2 values, found: {len(v['resolution'])}")

            if 'depth' in k:
                self._config.enable_stream(rs.stream.depth, v['resolution'][0], v['resolution'][1], rs.format.z16, v['fps'])
            elif 'color' in k:
                self._config.enable_stream(rs.stream.color, v['resolution'][0], v['resolution'][1], rs.format.bgr8, v['fps'])
            else:
                raise ValueError(f'No other camera type supported.')

    def _get_user_input(self):
        while True:
            input_val = input('Please insert if the programm should [e]xit, or the video settings should be [a]ligned. \n')
            cmd = input_val.strip().lower()

            if cmd == 'e':
                sys.exit()

            elif cmd == 'a':
                self._sensors_to_setup['color']['resolution'] = self._sensors_to_setup['depth']['resolution']
                self._sensors_to_setup['color']['fps'] = self._sensors_to_setup['depth']['fps']
                break
            
            else:
                self.logger.error('Wrong command.')
                continue

    def _warm_up(self):
        for _ in range(10):
            self._pipeline.wait_for_frames(timeout_ms=5000)
    
    def _mode_guard(self):
        if self._buf:
            raise RuntimeError("Can't call this method, while using the buffer.")