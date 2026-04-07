import cv2
import copy
import numpy as np
from dataclasses import dataclass
from camera.camera import Camera


@dataclass
class Rotation:
    """Create rotation matrix with degrees rotating around each axis."""

    alpha: float
    "Rotation around x-axis in unit degree"
    beta: float
    "Rotation around y-axis in unit degree"
    gamma: float
    "Rotation around z-axis in unit degree"

    def __post_init__(self):

        # convert to radian
        alpha = np.deg2rad(self.alpha)
        beta  = np.deg2rad(self.beta)
        gamma = np.deg2rad(self.gamma)

        # Rotation around X-axis
        self.Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])

        # Rotation around Y-axis
        self.Ry = np.array([
            [ np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        # Rotation around Z-axis
        self.Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        self.R = self.Rz @ self.Ry @ self.Rx


    def rotate_camera(self, src_camera: Camera):
        """Rotate a given source camera to get a new camera."""

        tgt_camera = copy.deepcopy(src_camera)
        tgt_camera.extrinsics.rotation = self.R @ src_camera.extrinsics.rotation
        return tgt_camera


@dataclass
class Projector:
    """Project images from source camera to target camera."""

    src_cam: Camera
    """From which camera to project the image."""
    tgt_cam: Camera
    """To which camera to project the image."""

    def __post_init__(self):
        """
        NOTE: remap requires backward mapping, 
        i.e. for the target pixels, where should Ithey sample in the source image.
        """

        # coordinates
        # NOTE: note the order of W and H
        u, v = np.meshgrid(
            np.arange(self.tgt_cam.width),
            np.arange(self.tgt_cam.height),
        )
        tgt_coords = np.stack((u, v, np.ones_like(u)),axis=-1).reshape(-1,3) # shape: [N, 3]

        # project 2d coordinates to 3d rays
        tgt_K_inv = np.linalg.inv(self.tgt_cam.intrinsics.matrix) # shape: [3, 3]
        tgt_rays = tgt_K_inv @ tgt_coords.T # shape: [3, N] = [3, 3] @ [N, 3].T

        # rotate source camera (i.e. rays) to target pose
        # compute the relative rotation:
        #   R_rel @ P_tgt = P_src
        #   R_rel @ (P_tgt @ P_w) = P_src @ P_w
        #   R_rel @ P_tgt = P_src
        #   R_rel = P_src @ inv(P_tgt)
        rotation_relative = self.src_cam.extrinsics.rotation @ np.linalg.inv(self.tgt_cam.extrinsics.rotation) # shape: [3, 3]
        src_rays = rotation_relative @ tgt_rays # shape: [3, N] = [3, 3] @ [3, N]
        
        # project 3d rays in target pose to 2d image space
        src_coords = (self.src_cam.intrinsics.matrix @ src_rays).T # shape: [N, 3] = ([3, 3] @ [3, N]).T

        # normalize
        # NOTE: "normalize after project" is the same with "project after normalize"
        src_coords /= src_coords[:, 2].reshape(-1, 1) # shape: [N, 2]

        # remapping
        # TODO: what if the target image size is different from source camera?
        # NOTE: note the order of H and W
        self.x_map = src_coords[:, 0].reshape(self.src_cam.height, self.src_cam.width).astype(np.float32) # x coordinate mapping, shape: [w, h]
        self.y_map = src_coords[:, 1].reshape(self.src_cam.height, self.src_cam.width).astype(np.float32) # y coordinate mapping, shape: [w, h]


    def __call__(self, img: np.ndarray):

        return cv2.remap(
            src=img, 
            map1=self.x_map, 
            map2=self.y_map, 
            interpolation=cv2.INTER_LINEAR,
        )