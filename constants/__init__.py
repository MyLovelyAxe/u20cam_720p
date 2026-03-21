import numpy as np
from pathlib import Path

REPO_ROOT = Path.home() / "u20cam_720p"
DEFAULT_DATA_TYPE = np.float32

# device

U20CAM_720P_SOURCE = "/dev/video2"   # same device as your test script

# calibration

SAVE_DIR_CALIB_IMAGES = REPO_ROOT / "calibration_images"
NUM_IMAGE_TO_SAVE = 25
INTERVAL_SECOND_TO_SAVE = 1.0
CALIB_PARAM_JSON = REPO_ROOT / "configs/u20cam_calib.json"