from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

src = Path("data/debug_live_crops")
files = sorted(src.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]

if not files:
    print("No debug crop JPGs found.")
else:
    cell_w, cell_h = 300, 150
    cols = 3
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(files):
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell_w - 12, cell_h - 35))
            x = (i % cols) * cell_w
            y = (i // cols) * cell_h
            sheet.paste(im, (x + (cell_w - im.width)//2, y + 4))
            draw.text((x + 5, y + cell_h - 25), path.name[:42], fill="black")
        except Exception as e:
            print(f"Could not open {path.name}: {e}")

    out = Path("data/debug_crops_contact_sheet.jpg")
    sheet.save(out)
    print(f"Saved: {out}")
