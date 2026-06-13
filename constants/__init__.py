import numpy as np
from pathlib import Path
from enum import Enum

###### Path ######

REPO_ROOT = Path.home() / "u20cam_720p"


###### Type ######

DEFAULT_DATA_TYPE = np.float32


###### Device ######

U20CAM_720P_SOURCE = "/dev/video2"   # same device as your test script
DEV_V4L_BY_PATH = "/dev/v4l/by-path"
LAPTOP_LEFT_USB_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:1:1.0-video-index0"
LAPTOP_RIGHT_USB_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2:1.0-video-index0"
# usb hub: connect to laptop right usb 1
# NOTE: socket ID 1-4 starting from the indicator light on the usb hub
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_1 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.1:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_2 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.2:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_3 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.3:1.0-video-index0"
LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4 = f"{DEV_V4L_BY_PATH}/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0"

class So101Camera(str, Enum):
    """The usb path of camera on usb hub for SO101 robot arm system."""
    wrist_cam = LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_3
    side_cam = LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4


###### Calibration ######

SAVE_DIR_CALIB_IMAGES = REPO_ROOT / "calibration/calibration_images"
NUM_IMAGE_TO_SAVE = 25
INTERVAL_SECOND_TO_SAVE = 1.0
CALIB_PARAM_JSON = REPO_ROOT / "configs/u20cam_calib.json"