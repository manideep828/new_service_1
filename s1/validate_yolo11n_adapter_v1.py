from pathlib import Path
import cv2
from s1.yolo_detector import YoloDetector

video = Path("data/uploads/b23f4810477643dabc89197ee13bb34a.mp4")
outdir = Path("data/yolo11n_adapter_validation")
outdir.mkdir(parents=True, exist_ok=True)

if not video.exists():
    raise FileNotFoundError(f"Video not found: {video.resolve()}")

detector = YoloDetector(
    model_path="yolo11n.pt",
    input_size=640,
    score_thr=0.15,
    nms_thr=0.45,
    classes={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
    threads=2,
)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {video}")
print(f"FPS: {fps:.2f}; frames: {total}; duration: {total / fps:.1f}s")

for sec in [10, 14, 16, 18, 20]:
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, frame = cap.read()
    if not ok:
        print(f"{sec}s: unable to read frame")
        continue

    dets = detector.detect(frame)
    print(f"\n{sec}s: {len(dets)} detections")
    for x1, y1, x2, y2, conf, cls_id in dets:
        label = f"class={int(cls_id)} conf={conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(frame, label, (int(x1), max(25, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"  class={int(cls_id)} conf={conf:.3f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    target = outdir / f"frame_{sec}s.jpg"
    if not cv2.imwrite(str(target), frame):
        raise RuntimeError(f"Could not save {target}")
    print(f"  saved: {target}")

cap.release()
print(f"\nValidation images: {outdir.resolve()}")
