import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
from camera.camera import U20Camera
from constants import (
    CALIB_PARAM_JSON,
    So101Camera,
)

wrist_cam = U20Camera(
    usb_port=So101Camera.wrist_cam,
    calibration_json=CALIB_PARAM_JSON,
)
side_cam = U20Camera(
    usb_port=So101Camera.side_cam,
    calibration_json=CALIB_PARAM_JSON,
)

print(f"Wrist cam intrinsics: {wrist_cam.intrinsics}")
print(f"Side cam intrinsics: {side_cam.intrinsics}")

try:

    count = 0
    while True:
        wrist_frame = wrist_cam.get_frame()
        side_frame = side_cam.get_frame()

        if wrist_frame is None:
            warnings.warn("No wrist frame is returned")
        if side_frame is None:
            warnings.warn("No side frame is returned")

        if wrist_frame is None or side_frame is None:
            count += 1
            continue

        if count % 100 == 0:
            logging.info(msg=f"The {count}th frame — wrist: {wrist_frame.shape}")
            logging.info(msg=f"The {count}th frame — side: {side_frame.shape}")

        cv2.imshow("Wrist Cam", wrist_frame)
        cv2.imshow("Side Cam", side_frame)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    
    wrist_cam.release_capture()
    logging.info("Turn off wrist camera.")
    side_cam.release_capture()
    logging.info("Turn off side camera.")
    cv2.destroyAllWindows()