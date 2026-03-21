from constants import (
    CALIB_PARAM_JSON,
)
from camera.camera import Camera, U20Camera

u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)
print(u20cam.intrinsics)
count = 0
while count < 30:
    frame = u20cam.get_frame()
    if frame:
        print(f"frame shape: {frame.shape}")
    count +=1