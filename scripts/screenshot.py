import cv2
import time
import os
from pathlib import Path

# ---- config ----
camera_source = "/dev/video2"   # same device as your test script
save_dir = Path.home() / "isaacsim_vla_ws/u20cam_720p/calibration_images"
num_images = 25
interval_s = 1.0
# ----------------

os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture(camera_source)
if not cap.isOpened():
    raise RuntimeError("Camera not opened")

# Schedule the first capture 1s after the first frame arrives
shots_taken = 0
next_shot_time = time.monotonic() + interval_s

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {shots_taken} failed; stopping early")
        break

    now = time.monotonic()
    if shots_taken < num_images and now >= next_shot_time:
        filename = os.path.join(save_dir, f"image_{shots_taken:02d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        shots_taken += 1
        next_shot_time += interval_s

        if shots_taken >= num_images:
            print("Enough images saved.")
            break

    cv2.imshow("Camera Test", frame)
    # waitKey keeps the window responsive without blocking long enough to miss captures
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done; camera released.")
