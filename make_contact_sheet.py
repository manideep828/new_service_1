import cv2
import math
import numpy as np
from pathlib import Path

folder = Path("data/debug_live_crops")
files = sorted(
    folder.glob("*.jpg"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)[:24]

if not files:
    raise SystemExit("No JPG crops found.")

thumbs = []
for path in files:
    img = cv2.imread(str(path))
    if img is None:
        continue

    h, w = img.shape[:2]
    scale = min(320 / w, 220 / h)
    img = cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale)))
    )

    canvas = np.full((255, 340, 3), 255, dtype=np.uint8)
    y = 25 + (220 - img.shape[0]) // 2
    x = (340 - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(canvas, path.name[:42], (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    thumbs.append(canvas)

cols = 3
rows = math.ceil(len(thumbs) / cols)
sheet = np.full((rows * 255, cols * 340, 3), 255, dtype=np.uint8)

for i, img in enumerate(thumbs):
    sheet[(i // cols) * 255:(i // cols + 1) * 255,
          (i % cols) * 340:(i % cols + 1) * 340] = img

out = folder / "contact_sheet.jpg"
cv2.imwrite(str(out), sheet)
print(f"Saved: {out.resolve()} | crops: {len(thumbs)}")
