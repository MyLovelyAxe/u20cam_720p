import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from threading import Thread, Lock, Event

from camera.camera import UsbCamera
from constants import (
    DEFAULT_DATA_TYPE,
    U20CAM_720P_SOURCE,
    LAPTOP_LEFT_USB_1,
    LAPTOP_RIGHT_USB_1,
    CALIB_PARAM_JSON,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_1,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_2,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_3,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4,
)



@dataclass
class BaseCameraReader:
    """Template of frame readers which read the frames from multiple cameras."""

    cameras: Dict[str, UsbCamera]
    """Multiple cameras which captures live-stream frames, i.e. name: camera"""


    def start(self):
        """Connect all cameras."""
        for name, camera in self.cameras.items():
            logging.info(f"Connecting camera {name} at port {camera.usb_port}...")
            camera.connect()
                

    def stop(self):
        """Disconnect all cameras."""
        for name, camera in self.cameras.items():
            logging.info(f"Disconnecting camera {name} at port {camera.usb_port}...")
            camera.disconnect()


    def get_frames(self) -> Dict[str, np.ndarray | None]:
        """Read the latest frames from cameras."""
        raise NotImplementedError



@dataclass
class SyncCameraReader(BaseCameraReader):
    """Synchronously reads the frames from multiple cameras.
    
    This reader always reads the latest frames from all the cameras, take them
    as the frames from the same timestamps or timestamps close enough.
    """

    def get_frames(self) -> Dict[str, np.ndarray | None]:

        frames = dict()
        for name, camera in self.cameras.items():
            frame = camera.latest_frame
            if frame is not None:
                frames[name] = frame
            else:
                frames[name] = None
        return frames

