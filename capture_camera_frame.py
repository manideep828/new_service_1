import getpass
from pathlib import Path
import cv2

source = getpass.getpass("Enter CCTV RTSP URL (hidden): ")
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

try:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("FAILED: Could not read a frame. Check the RTSP URL, network, and camera stream path.")
    else:
        h, w = frame.shape[:2]
        out = Path("data") / "camera_reference.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(out), frame):
            print(f"SUCCESS: Saved {out.resolve()}")
            print(f"Actual camera frame: width={w}, height={h}")
        else:
            print("FAILED: Could not save the image.")
finally:
    cap.release()
