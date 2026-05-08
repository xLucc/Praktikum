import time
import sys
import logging
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from pathlib import Path
import json
from abc import ABC, abstractmethod
from typing import Optional
from bin_picking.camera.camera_interface import Camera
from bin_picking.common.helper import load_dict_from_json, get_project_dir


class RealSenseCamera(Camera):

    def __init__(self, serial: Optional[str]='', adv: Optional[str]='', align: Optional[bool]=True):
        
        self._ctx = rs.context()
        self._dual = align

        if align:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        if serial:
            self._serial = serial
        else:
            self._get_serial()

        self.logger = logging.getLogger(f'RealSense_{self._serial}')

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._product_line = self._get_line()
        self._config.enable_device(self._serial)

        self._root = get_project_dir()
        self._data_dir = self._root / 'data' 
        self._config_path = self._data_dir / 'realsense_config'
        self._config_path.mkdir(parents=True, exist_ok=True)
        self._setup_cfg = self._config_path / 'setup_cfg.json'

        if not self._setup_cfg.exists():
            raise FileNotFoundError('No setup file found.')

        if adv:
            if not adv.endswith('.json'):
                raise RuntimeError(f'Wrong config file type. Needs to be json')
            
            self._adv_settings = self._config_path / adv
            self.logger.info('Found advanced settings.')
            self._set_adv()

        self._sensors_to_setup = load_dict_from_json(self._setup_cfg)
        self._setup_sensors()

        self.logger.info('Done with initialisation.')
        self._profile = self._pipeline.start(self._config)
        self._warm_up()

    def __del__(self):
        self.logger.info('Clean up.')

        if hasattr(self, '_profile'):
            self._pipeline.stop()
        

    def stop(self):
        self.logger.info('Stop the pipeline.')
        self._pipeline.stop()

    def get_depth(self, num_frames: int=1):
        return self._get_data(num_frames=num_frames, mode='depth')

    def get_rgb(self, num_frames: int=1):
        return self._get_data(num_frames=num_frames, mode='color')


    def stream(self) ->np.ndarray:

        '''
        Streams the color image.
        
        Returns:
            img (np.ndarray): If wanted return the last img.

        Raises:
            KeyboardInterrupt: To exit the stream.
        '''

        self.logger.info('To save the image, press s. \n To quit the stream, press q.')

        while True:

            frame = self._pipeline.wait_for_frames().get_color_frame()

            if not frame:
                continue
            

            img = np.asanyarray(frame.get_data())
            cv.imshow('img', img)
            key = cv.waitKey(1) & 0xFF

            if key == ord("s"):
                cv.destroyAllWindows()
                return img
            elif key == ord("q"):
                cv.destroyAllWindows()
                raise KeyboardInterrupt

    

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
                        print(f'Device {serial} is busy.')
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

        # Only avialable for D400 Series.
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
        for _ in range(20):
            self._pipeline.wait_for_frames(timeout_ms=10000)
