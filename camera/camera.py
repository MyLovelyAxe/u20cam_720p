import cv2
import json
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from constants import (
    DEFAULT_DATA_TYPE,
    U20CAM_720P_SOURCE,
    CALIB_PARAM_JSON,
)


@dataclass
class Intrinsics:

    fx: float
    """Focal length along with x direction."""
    fy: float
    """Focal length along with y direction."""
    cx: float
    """X coordinate of principal point."""
    cy: float
    """Y coordinate of principal point."""
    data_type: type = DEFAULT_DATA_TYPE
    """Basic data type for np.array-type attributes."""

    @property
    def focal_length(self) -> Tuple[float]:
        return (self.fx, self.fy)
    
    @property
    def principal_point(self) -> Tuple[float]:
        return (self.cx, self.cy)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx,   0,          self.cx ],
                [0,         self.fy,    self.cy ],
                [0,         0,          1       ],
            ],
            dtype=self.data_type,
        )

    @classmethod
    def create_from_matrix(cls, matrix: np.ndarray) -> 'Intrinsics':
        """Create Intrinsics object with matrix."""
        return cls(
            fx=float(matrix[0,0]),
            fy=float(matrix[1,1]),
            cx=float(matrix[0,2]),
            cy=float(matrix[1,2]),
        )



@dataclass
class Extrinsics:

    rotation: np.ndarray
    """Rotation matrix with shape (3, 3)."""
    translation: np.ndarray
    """Translation vector with shape (3,)"""
    data_type: type = DEFAULT_DATA_TYPE
    """Basic data type for np.array-type attributes."""

    @property
    def homogenous(self) -> np.ndarray:
        hom = np.identity(4, dtype=self.data_type)
        hom[0:3, 0:3] = self.rotation
        hom[0:3, 3] = self.translation
        return hom
    
    @classmethod
    def create_from_homogeneous(cls, hom: np.ndarray) -> 'Intrinsics':
        """Create Intrinsics object with matrix."""
        assert hom.shape == (4,4), f"The homogeneous matrix shape is {hom.shape}, not (4,4)"
        return cls(
            rotation=hom[0:3, 0:3],
            translation=hom[0:3, 3].reshape(-1),
        )



@dataclass
class Camera:
    """Base class for all type of cameras."""

    intrinsics: Intrinsics
    """Including focal lengths, principal points, distorsion coefficients."""
    extrinsics: Optional[Extrinsics]
    """Including rotation and translation."""
    dist_coeffs: np.ndarray
    """Distorsion coefficients, shape (length,)"""
    width: float
    """Width of raw image."""
    height: float
    """Height of raw image."""
    model: str = "opencv_pinhole"
    """The model of this camera."""
    alpha: float = 0.0
    """TODO: clear documentation here, when 1.0: keep full FoV, large distortion at edge"""
    data_type: type = DEFAULT_DATA_TYPE
    """Basic data type for np.array-type attributes."""
    calibration_json: Optional[Path] = None
    """The u20cam_calib.json containing the camera calibration parameters."""

    def __post_init__(self):
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.intrinsics.matrix,
            distCoeffs=self.dist_coeffs,
            imageSize=(self.width, self.height),
            alpha=self.alpha,   
            newImgSize=(self.width, self.height),
        )
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi

    @classmethod
    def create_from_json(cls, calib_json: Path):
        """Create a Camera object given json file with calibration parameters."""
        with open(calib_json, "r") as f:
            calib = json.load(f)
        return cls(
            intrinsics=Intrinsics(
                fx=calib["intrinsics"]["fx"],
                fy=calib["intrinsics"]["fy"],
                cx=calib["intrinsics"]["cx"],
                cy=calib["intrinsics"]["cy"],
            ),
            extrinsics=None,
            dist_coeffs=np.array(
                calib["distortion"]["coefficients"], 
                dtype=cls.data_type,
            ),
            width=calib["image_width"],
            height=calib["image_height"],
            model=calib["distortion"]["model"],
            calibration_json=calib_json,
        )

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort the current frame."""

        # Undistort
        undistorted = cv2.undistort(
            src=frame,
            cameraMatrix=self.intrinsics.matrix,
            distCoeffs=self.dist_coeffs,
            dst=None,
            newCameraMatrix=self.new_camera_matrix
        )

        # Crop and resize to original size
        cropped = undistorted[
            self.roi_y:self.roi_y+self.roi_h, 
            self.roi_x:self.roi_x+self.roi_w,
        ]
        resized_undistorted = cv2.resize(
            src=cropped, 
            dsize=(self.width, self.height), 
            interpolation=cv2.INTER_LINEAR,
        )
        return resized_undistorted


@dataclass
class U20Camera(Camera):
    """Camera model of U20CAM 720P."""

    @property
    def source_device(self) -> str:
        "The device name of the camera, either given or use the default one."
        return U20CAM_720P_SOURCE

    @property
    def camera_connected(self) -> bool:
        "True: the camera is connected; False: camera is not connected."
        return self.capture.isOpened()

    def __post_init__(self):
        self.capture = cv2.VideoCapture(self.source_device)

    def get_frame(self, undistort: bool = True) -> np.ndarray | None:
        """Get a frame from a running capture."""
        if not self.camera_connected:
            warnings.warn("The camera is not connected, no running capture.")
            return None
        else:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                return None
            if undistort:
                return self.undistort_frame(frame)
            else:
                return frame



if __name__ == "__main__":

    u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)
    print(u20cam.intrinsics)
    count = 0
    while count < 30:
        frame = u20cam.get_frame()
        if frame:
            print(f"frame shape: {frame.shape}")
        count +=1