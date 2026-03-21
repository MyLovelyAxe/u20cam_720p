from constants import (
    CALIB_PARAM_JSON,
)
from camera.camera import Camera, U20Camera

cam = Camera.create_from_json(CALIB_PARAM_JSON)
u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)
print(cam.intrinsics)