import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from threading import Thread, Lock, Event

from constants import (
    DEFAULT_DATA_TYPE,
    CALIB_PARAM_JSON,
    LAPTOP_LEFT_USB_1,
    LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4,
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
    def create_from_homogeneous(cls, hom: np.ndarray) -> 'Extrinsics':
        """Create Intrinsics object with matrix."""

        assert hom.shape == (4,4), f"The homogeneous matrix shape is {hom.shape}, not (4,4)"
        return cls(
            rotation=hom[0:3, 0:3],
            translation=hom[0:3, 3].reshape(-1),
        )



@dataclass
class Camera:
    """Base class for all type of cameras."""

    intrinsics: Intrinsics = None
    """Including focal lengths, principal points, distorsion coefficients."""
    extrinsics: Optional[Extrinsics] = None
    """Including rotation and translation."""
    dist_coeffs: np.ndarray = None
    """Distorsion coefficients, shape (length,)"""
    width: float = None
    """Width of raw image."""
    height: float = None
    """Height of raw image."""
    model: str = "opencv_pinhole"
    """The model of this camera."""
    alpha: float = 0.0
    """When 1.0: keep full FoV, large distortion at edge"""
    data_type: type = DEFAULT_DATA_TYPE
    """Basic data type for np.array-type attributes."""
    calibration_json: Optional[Path] = None
    """The u20cam_calib.json containing the camera calibration parameters."""
    calib_loaded: bool = False
    """Indicator whether the calibration parameters are already loaded."""


    def __post_init__(self):

        # Load the camera
        if not self.calib_loaded:
            self.load_calib_params_from_json(self.calibration_json)
        self.get_optimal_new_camera_matrix()


    def get_optimal_new_camera_matrix(self):
        """Compute the new intrinsics matrix and ROI for rectifying distorted raw image."""
        
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.intrinsics.matrix,
            distCoeffs=self.dist_coeffs,
            imageSize=(self.width, self.height),
            alpha=self.alpha,   
            newImgSize=(self.width, self.height),
        )
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi


    def load_calib_params_from_json(self, calib_json: Path):
        """Load calibration parameters from exteral json."""

        with open(calib_json, "r") as f:
            calib = json.load(f)
        self.intrinsics=Intrinsics(
            fx=calib["intrinsics"]["fx"],
            fy=calib["intrinsics"]["fy"],
            cx=calib["intrinsics"]["cx"],
            cy=calib["intrinsics"]["cy"],
        )
        self.extrinsics=None
        self.dist_coeffs=np.array(
            calib["distortion"]["coefficients"], 
            dtype=self.data_type,
        )
        self.width=calib["image_width"]
        self.height=calib["image_height"]
        self.model=calib["distortion"]["model"]
        self.calibration_json=calib_json
        self.calib_loaded = True


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
            calib_loaded=True,
        )


    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort the current frame."""

        # Undistort
        undistorted = cv2.undistort(
            src=frame,
            cameraMatrix=self.intrinsics.matrix,
            distCoeffs=self.dist_coeffs,
            dst=None,
            newCameraMatrix=self.new_camera_matrix,
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
class UsbCamera(Camera):
    """Cameras which needs to be physically connected by usb port and stream."""

    usb_port: str = LAPTOP_LEFT_USB_1
    """The device name of the camera."""
    undistort: bool = True
    """Whether undirtort the captured frame or not."""


    @property
    def is_connected(self) -> bool:
        "True: the camera is connected; False: camera is not connected."
        return self.capture is not None and self.capture.isOpened()
    
    @property
    def latest_frame(self) -> np.ndarray | None:
        """Read the buffer for the latest captured and stored frame."""
        with self._lock:
            return self._buffer


    def __post_init__(self):

        super().__post_init__()
        self.capture = None
        self._camera_thread = None
        self._lock = Lock()
        self._stop_event = Event() # Stop signal to stop the capture loop
        self._has_frame_event = Event() # Set when the first frame is ready
        self._buffer: Optional[np.ndarray] = None # Store the latest frames


    def _open_camera(self):
        """Create capture to be ready for frame capturing."""

        self.capture = cv2.VideoCapture(self.usb_port)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Failed to open camera at port {self.usb_port}")
        logging.info(f"Camera at port {self.usb_port} is opened.")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    def connect(self):
        """Connect camera and start the capturing thread."""

        self._stop_event.clear()
        self._has_frame_event.clear()
        self._camera_thread = Thread(
            target=self._capture_loop,
            daemon=True, # NOTE: This thread should not prevent the Python program from exiting
        )
        self._camera_thread.start()
        self._has_frame_event.wait() # wait for the first frame to be ready, then return, finish connection

    
    def _capture_loop(self):
        """Capture a frame and store in the buffer."""

        self._open_camera()

        try:
            while not self._stop_event.is_set():
                if not self.is_connected:
                    warnings.warn(
                        f"The camera is not connected, stopping capture loop."
                    )
                    break
                ret, frame = self.capture.read()
                if not ret:
                    warnings.warn("Failed to grab frame")
                    continue
                if self.undistort:
                    frame = self.undistort_frame(frame)
                with self._lock:
                    self._buffer = frame
                if not self._has_frame_event.is_set():
                    self._has_frame_event.set()
        finally:
            self.release_capture()


    def disconnect(self):
        """Set stop signal for stopping capture loop."""

        self._stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join()
            self._camera_thread = None
        self._has_frame_event.clear()
        logging.info(f"The camera is disconnected")


    def release_capture(self):
        """Stop streaming images."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None
        logging.info(f"The capture is released")



@dataclass
class U20Camera(UsbCamera):
    """Camera model of U20CAM 720P."""


if __name__ == "__main__":

    # u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)
    u20cam = U20Camera(
        usb_port=LAPTOP_RIGHT_USB_1_USB_HUB_SOCKET_4,
        calibration_json=CALIB_PARAM_JSON,
    )
    print(u20cam.intrinsics)

    try:
        count = 0
        u20cam.connect()
        while True:
            frame = u20cam.latest_frame
            if frame is None:
                warnings.warn(f"No frame is returned, skip...")
                count +=1
                continue

            if count % 100 == 0:
                logging.info(f"{count}th frame shape: {frame.shape}")
            cv2.imshow(f"Camera at port {u20cam.usb_port}", frame)

            count +=1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        u20cam.disconnect()