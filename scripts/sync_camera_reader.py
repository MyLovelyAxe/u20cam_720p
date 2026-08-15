"""
This script shows how to capture synchronous frames from multiple cameras.

Usage:
    $ conda activate u20cam
    $ cd ~/u20cam_720p/scripts
    $ python sync_camera_reader.py
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
from camera_utils.camera import U20Camera

from constants import (
    DEFAULT_DATA_TYPE,
    CALIB_PARAM_JSON,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_2,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4,
)
from camera_utils.reader import SyncCameraReader
from camera_utils.image import Frame


def main():

    camera_2 = U20Camera(
        usb_port=LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_2,
        calibration_json=CALIB_PARAM_JSON,
    )
    camera_4 = U20Camera(
        usb_port=LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4,
        calibration_json=CALIB_PARAM_JSON,
    )
    reader = SyncCameraReader(
        cameras=dict(
            camera_2=camera_2,
            camera_4=camera_4,
        )
    )

    try:

        reader.start()
        frames = None

        count = 0

        empty_frame = np.zeros((480, 640, 3), dtype=DEFAULT_DATA_TYPE)

        while True:
            frames = reader.get_frames()
            for camera_name, frame in frames.items():
                if frame is None:
                   logging.warning(f"Camera {camera_name} failed to capture frame, show empty frame")
                   frame = Frame(frame=np.copy(empty_frame))
                cv2.imshow(f"Camera {camera_name}", frame.frame)

            if count % 100 == 0:
                log_str = f"{count}th frames: "
                for camera_name, frame in frames.items():
                    log_str += f"Camera {camera_name} timestamp: {frame.timestamp}, time: {frame.time}, shape: {frame.frame.shape} | "
                logging.info(log_str)

            count +=1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        reader.stop()


if __name__ == "__main__":

    main()