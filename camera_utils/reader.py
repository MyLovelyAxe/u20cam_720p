import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import numpy as np
from dataclasses import dataclass
from typing import Dict

from camera_utils.camera import UsbCamera
from camera_utils.image import Frame



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

    def get_frames(self) -> Dict[str, Frame | None]:

        frames = dict()
        for name, camera in self.cameras.items():
            frame = camera.latest_frame
            if frame is not None:
                frames[name] = frame
            else:
                frames[name] = None
        return frames

