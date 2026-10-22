import numpy as np
from ultralytics import YOLO


def drop_contained_boxes(dets, frac=0.85):
    """Drop a box when >= frac of its area lies inside a larger box of the same class."""
    n = len(dets)
    if n < 2:
        return dets
    area = (dets[:, 2] - dets[:, 0]).clip(0) * (dets[:, 3] - dets[:, 1]).clip(0)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j or dets[i, 5] != dets[j, 5] or area[j] <= area[i]:
                continue
            iw = min(dets[i, 2], dets[j, 2]) - max(dets[i, 0], dets[j, 0])
            ih = min(dets[i, 3], dets[j, 3]) - max(dets[i, 1], dets[j, 1])
            if iw > 0 and ih > 0 and (iw * ih) / max(area[i], 1e-6) >= frac:
                keep[i] = False
                break
    return dets[keep]

class YoloDetector:
    CONTAINED_FRAC = 0.85

    """YOLO11 detector with the existing (N, 6) pipeline contract.

    Returns rows: x1, y1, x2, y2, confidence, COCO class ID.
    Coordinates are in original-frame pixels.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        input_size=640,
        score_thr=0.15,
        nms_thr=0.45,
        classes=None,
        threads=2,
    ):
        self.size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.allowed = sorted(
            (classes or {
                2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
            }).keys()
        )
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, img):
        result = self.model.predict(
            source=img,
            imgsz=self.size,
            conf=self.score_thr,
            iou=self.nms_thr,
            classes=self.allowed,
            verbose=False,
            device="cpu",
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)

        out = np.concatenate((xyxy, conf, cls), axis=1).astype(np.float32)
        return drop_contained_boxes(out, self.CONTAINED_FRAC)
