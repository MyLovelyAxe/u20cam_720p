#!/usr/bin/env python3
"""
Calibrate a camera from all .png images under a given directory using OpenCV,
and save the intrinsics + distortion coefficients to u20cam_calib.json.

Usage example:
    python calibrate_camera.py \
        --image_dir /path/to/calib_images \
        --rows 6 \
        --cols 9 \
        --square_size 0.025 \
        --output u20cam_calib.json

Notes:
- rows / cols are the number of INNER corners, not squares.
- square_size is the checkerboard square size in meters (or any unit you want).
  For intrinsics only, the absolute unit is not critical, but keep it consistent.
- This script assumes a normal distorted pinhole camera model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera from checkerboard PNG images.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/hardli/isaacsim_vla_ws/u20cam_720p/calibration_images",
        help="Directory containing calibration .png images",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of inner corners along checkerboard rows",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of inner corners along checkerboard cols",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=0.025,
        help="Checkerboard square size in real-world units, e.g. 0.025 for 25 mm",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="u20cam_calib.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .png files under image_dir",
    )
    parser.add_argument(
        "--show_corners",
        action="store_true",
        help="Show detected checkerboard corners while processing",
    )
    return parser.parse_args()


def collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    images = sorted(image_dir.glob(pattern))
    return [p for p in images if p.is_file()]


def build_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    """
    Build checkerboard 3D points on z=0 plane.
    Shape: (rows*cols, 3)
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = collect_images(image_dir, args.recursive)
    if not images:
        raise FileNotFoundError(f"No .png images found in: {image_dir}")

    pattern_size = (args.cols, args.rows)  # OpenCV expects (cols, rows)
    objp = build_object_points(args.rows, args.cols, args.square_size)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    image_size: tuple[int, int] | None = None
    valid_images: list[str] = []
    failed_images: list[str] = []

    # Optional refinement criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            failed_images.append(str(img_path))
            print(f"[WARN] Could not read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        else:
            current_size = (gray.shape[1], gray.shape[0])
            if current_size != image_size:
                failed_images.append(str(img_path))
                print(
                    f"[WARN] Skipping {img_path} because image size {current_size} "
                    f"does not match first image size {image_size}"
                )
                continue

        # Try to find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if not found:
            failed_images.append(str(img_path))
            print(f"[WARN] Checkerboard not found: {img_path}")
            continue

        # Refine corner positions
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners_subpix)
        valid_images.append(str(img_path))
        print(f"[OK] Checkerboard detected: {img_path}")

        if args.show_corners:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_subpix, found)
            cv2.imshow("Corners", vis)
            key = cv2.waitKey(300)
            if key == 27:  # ESC
                break

    if args.show_corners:
        cv2.destroyAllWindows()

    if not objpoints or image_size is None:
        raise RuntimeError("No valid checkerboard detections found. Cannot calibrate camera.")

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    # Compute reprojection error
    total_error = 0.0
    per_image_errors: list[float] = []
    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
        per_image_errors.append(float(error))
        total_error += error

    mean_error = total_error / len(objpoints)

    # Extract common intrinsic parameters
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    # Distortion coefficients are typically [k1, k2, p1, p2, k3, ...]
    dist_list = dist_coeffs.flatten().tolist()

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "intrinsics": {
            "camera_matrix": camera_matrix.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "distortion": {
            "coefficients": dist_list,
            "model": "opencv_pinhole",
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=4))
    print(f"\nCalibration saved to: {output_path.resolve()}")
    print(f"Images used: {len(valid_images)} / {len(images)}")
    print(f"RMS reprojection error: {ret:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"Distortion coefficients: {dist_list}")


if __name__ == "__main__":
    main()