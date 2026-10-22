"""Fit-to-window parking geometry calibration tool.

Clicks are mapped back to original camera-frame coordinates.
Keys: u=undo, n=skip incomplete bay, Enter/s=save, q=quit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="config/geometry.json")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--max-width", type=int, default=1100)
    ap.add_argument("--max-height", type=int, default=800)
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read a frame from source: {a.source}")

    H, W = img.shape[:2]
    scale = min(a.max_width / W, a.max_height / H, 1.0)
    disp_w = max(1, round(W * scale))
    disp_h = max(1, round(H * scale))
    preview = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    scale_x = W / disp_w
    scale_y = H / disp_h

    wire, inside, bays, cur = [], [], [], []
    message = ""

    def phase():
        if len(wire) < 2:
            return "1/3: Click 2 gate/tripwire points"
        if not inside:
            return "2/3: Click 1 point INSIDE the parking area"
        return f"3/3: Bay B-{len(bays)+1:02d}: click 4 corners"

    def on_mouse(event, x, y, flags, param):
        nonlocal message
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not (0 <= x < disp_w and 0 <= y < disp_h):
            return
        point = (min(W - 1, round(x * scale_x)),
                 min(H - 1, round(y * scale_y)))
        if len(wire) < 2:
            wire.append(point)
        elif not inside:
            inside.append(point)
        else:
            cur.append(point)
            if len(cur) == 4:
                bays.append(list(cur))
                cur.clear()
        message = ""

    window = "Parking Calibration - fit to screen"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        v = preview.copy()

        def dp(point):
            return (round(point[0] / scale_x), round(point[1] / scale_y))

        if len(wire) == 2:
            cv2.line(v, dp(wire[0]), dp(wire[1]), (0, 255, 255), 2)
        for point in wire + inside:
            cv2.circle(v, dp(point), 4, (0, 255, 255), -1)

        for i, bay in enumerate(bays):
            poly = np.array([dp(p) for p in bay], dtype=np.int32)
            cv2.polylines(v, [poly], True, (0, 200, 0), 2)
            cv2.putText(v, f"B-{i+1:02d}", tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        for point in cur:
            cv2.circle(v, dp(point), 4, (0, 0, 255), -1)

        cv2.rectangle(v, (0, 0), (disp_w, 52), (20, 20, 20), -1)
        cv2.putText(v, phase(), (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(v, f"Frame: {W}x{H} | Preview: {disp_w}x{disp_h} | u undo | n skip | s save | q quit",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (200, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, v)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("u"):
            if cur:
                cur.pop()
            elif bays:
                cur.extend(bays.pop())
            elif inside:
                inside.pop()
            elif wire:
                wire.pop()
        elif key == ord("n"):
            if cur:
                cur.clear()
                message = "Incomplete bay discarded"
        elif key in (13, 10, ord("s")):
            if len(wire) != 2 or not inside:
                print("Set both tripwire points and the inside point before saving.")
                continue
            if cur:
                print(f"Finish or discard the {len(cur)} incomplete bay points before saving.")
                continue

            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "frame_size": [W, H],
                "tripwire": wire,
                "inside_point": inside[0],
                "plate_zone": None,
                "slots": [
                    {"slot_id": f"B-{i+1:02d}", "level": a.level, "polygon": bay}
                    for i, bay in enumerate(bays)
                ],
            }
            with out.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {out}: frame {W}x{H}, {len(bays)} bays.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
