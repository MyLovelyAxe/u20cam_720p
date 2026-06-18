"""
This script launches multiple cameras.

$ conda activate u20cam
$ cd ~/u20cam_720p

Only launch wrist camera:
$ python multi_cameras.py --wrist

Only launch side camera:
$ python multi_cameras.py --sode

Launch both wrist and side cameras:
$ python multi_cameras.py --wrist --side
"""

import warnings
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import argparse
import cv2
from camera.camera import U20Camera
from constants import (
    CALIB_PARAM_JSON,
    So101Camera,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrist",
        action="store_true",
        default=False,
        help="Whether launch wrist camera",
    )
    parser.add_argument(
        "--side",
        action="store_true",
        default=False,
        help="Whether launch side camera",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    print(f"args.wrist: {args.wrist}")
    print(f"args.side: {args.side}")

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

        wrist_frame = None
        side_frame = None

        count = 0
        while True:
            if args.wrist:
                wrist_frame = wrist_cam.get_frame()
            if args.side:
                side_frame = side_cam.get_frame()

            skip = False
            if args.wrist and wrist_frame is None:
                warnings.warn("No wrist frame is returned")
                skip = True
            if args.side and side_frame is None:
                warnings.warn("No side frame is returned")
                skip = True

            if skip:
                count += 1
                continue

            if count % 100 == 0:
                if args.wrist:
                    logging.info(msg=f"The {count}th frame — wrist: {wrist_frame.shape}")
                if args.side:
                    logging.info(msg=f"The {count}th frame — side: {side_frame.shape}")

            if args.wrist:
                cv2.imshow("Wrist Cam", wrist_frame)
            if args.side:
                cv2.imshow("Side Cam", side_frame)

            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        
        if args.wrist:
            wrist_cam.release_capture()
            logging.info("Turn off wrist camera.")
        if args.side:
            side_cam.release_capture()
            logging.info("Turn off side camera.")
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()