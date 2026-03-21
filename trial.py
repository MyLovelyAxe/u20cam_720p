import cv2
from camera.camera import U20Camera
from constants import CALIB_PARAM_JSON


u20cam = U20Camera.create_from_json(CALIB_PARAM_JSON)
print(u20cam.intrinsics)
count = 0
while True:
    frame = u20cam.get_frame()
    if frame:
        print(f"frame shape: {frame.shape}")
    count +=1

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

u20cam.release_capture()
cv2.destroyAllWindows()