import cv2
import numpy as np
import json
from pathlib import Path
from constants import CALIB_PARAM_JSON, SAVE_DIR_CALIB_IMAGES

# ---- paths ----
ROOT_PATH = Path.home() / "isaacsim_vla_ws/u20cam_720p"
calib_json = ROOT_PATH / "u20cam_calib.json"
input_image = SAVE_DIR_CALIB_IMAGES / "05.png"
output_image = ROOT_PATH / "undistorted.png"

# ---- load calibration from json ----
with open(CALIB_PARAM_JSON, "r") as f:
    calib = json.load(f)

camera_matrix = np.array(calib["intrinsics"]["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(calib["distortion"]["coefficients"], dtype=np.float64)

# ---- read image ----
img = cv2.imread(input_image)
if img is None:
    raise RuntimeError(f"Cannot read image: {input_image}")

h, w = img.shape[:2]

# ---- compute optimal new camera matrix ----
# TODO: save the new_camera_matrix into new calib
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    # alpha=1.0,   # keep full FoV, large distortion at edge
    alpha=0.0,   
    newImgSize=(w, h)
)

# ---- undistort image ----
undistorted = cv2.undistort(
    img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

# ---- optional crop ----
x, y, roi_w, roi_h = roi
if roi_w > 0 and roi_h > 0:
    cropped = undistorted[y:y+roi_h, x:x+roi_w]
    resized_undistorted = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
print(f"undistorted image size: {resized_undistorted.shape}")

# ---- save result ----
cv2.imwrite(output_image, resized_undistorted)

# ---- display ----
cv2.imshow("raw", img)
cv2.imshow("undistorted without cropped", undistorted)
cv2.imshow("undistorted resized back to raw size", resized_undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()