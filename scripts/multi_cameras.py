"""
This script launches multiple cameras.

Usage:
    $ conda activate u20cam
    $ cd ~/u20cam_720p/scripts
    $ python multi_cameras.py
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

    try:

        camera_2.connect()
        camera_4.connect()
        frame_2 = None
        frame_4 = None

        count = 0

        empty_frame = np.zeros((480, 640, 3), dtype=DEFAULT_DATA_TYPE)

        while True:
            frame_2 = camera_2.latest_frame
            frame_4 = camera_4.latest_frame
            if frame_2 is None:
                logging.warning(f"No frame_2 is returned, show empty frame")
                frame_2 = Frame(frame=np.copy(empty_frame))
            if frame_4 is None:
                logging.warning(f"No frame_4 is returned, show empty frame")
                frame_2 = Frame(frame=np.copy(empty_frame))

            if count % 100 == 0:
                logging.info(
                    f"{count}th frame 2 time: {frame_2.time}, shape: {frame_2.frame.shape}, "
                    f"{count}th frame 4 time: {frame_4.time}, shape: {frame_4.frame.shape}"
                )
            cv2.imshow(f"Camera at port {camera_2.usb_port}", frame_2.frame)
            cv2.imshow(f"Camera at port {camera_4.usb_port}", frame_4.frame)

            count +=1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        camera_2.disconnect()
        camera_4.disconnect()


if __name__ == "__main__":

    main()