import cv2
import time
from pathlib import Path
from ultralytics import YOLO

VIDEO = "data/uploads/b23f4810477643dabc89197ee13bb34a.mp4"
MODEL = "yolo11n.pt"
OUT = Path("data/yolo11n_test")
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

for sec in [10, 14, 16, 18, 20]:
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, frame = cap.read()
    if not ok:
        print(f"{sec}s: frame unavailable")
        continue

    start = time.perf_counter()
    result = model.predict(
        source=frame,
        imgsz=640,
        conf=0.15,
        iou=0.45,
        classes=[2, 3, 5, 7],
        verbose=False,
        device="cpu"
    )[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = frame.copy()
    boxes = result.boxes
    print(f"\n=== {sec}s | {len(boxes)} detections | {elapsed_ms:.1f} ms ===")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {score:.2f}"
        print(f"  {label}: [{x1}, {y1}, {x2}, {y2}]")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(annotated, label, (x1, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    path = OUT / f"frame_{sec}s.jpg"
    if not cv2.imwrite(str(path), annotated):
        raise SystemExit(f"Could not save {path}")
    print("Saved:", path)

cap.release()
print("\nOffline test complete. No Service 2 events were sent.")
