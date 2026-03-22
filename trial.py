import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
from camera.camera import U20Camera, Camera
from constants import CALIB_PARAM_JSON

cam = Camera.create_from_json(CALIB_PARAM_JSON)
u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)

print(u20cam.intrinsics)
count = 0
while True:
    frame = u20cam.get_frame()
    if frame is None:
        warnings.warn("No frame is returned")
        continue
    else:
        if count % 100 == 0:
            logging.info(msg=f"The {count}th frame shape: {frame.shape}")
        cv2.imshow("Camera Test", frame)

    count +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

u20cam.release_capture()
cv2.destroyAllWindows()