import json
from pathlib import Path

path = Path("data/geometry_calibration_reordered.json")
with path.open(encoding="utf-8") as f:
    g = json.load(f)

W, H = g["frame_size"]
print(f"Frame: {W} x {H}")
print(f"Tripwire points: {len(g['tripwire'])}")
print(f"Slots found: {len(g.get('slots', []))}")

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def intersects(a, b, c, d):
    return orient(a,b,c)*orient(a,b,d) < 0 and orient(c,d,a)*orient(c,d,b) < 0

errors = []
ids = set()

for slot in g.get("slots", []):
    sid = slot.get("slot_id", "?")
    pts = slot.get("polygon", [])
    problems = []

    if sid in ids:
        problems.append("duplicate slot ID")
    ids.add(sid)

    if len(pts) != 4:
        problems.append(f"expected 4 corners, got {len(pts)}")
    else:
        for i, (x, y) in enumerate(pts):
            if not (0 <= x < W and 0 <= y < H):
                problems.append(f"corner {i+1} outside frame: {(x,y)}")

        if intersects(pts[0],pts[1],pts[2],pts[3]) or intersects(pts[1],pts[2],pts[3],pts[0]):
            problems.append("self-crossing polygon (X shape)")

        area = abs(sum(
            pts[i][0]*pts[(i+1)%4][1] - pts[(i+1)%4][0]*pts[i][1]
            for i in range(4)
        )) / 2
        if area < 100:
            problems.append(f"very small polygon area: {area:.1f} px^2")

    if problems:
        errors.append((sid, problems))
        print("CHECK:", sid, "|", "; ".join(problems))
    else:
        print("OK:", sid)

if errors:
    print(f"\nRESULT: {len(errors)} slot(s) need attention.")
else:
    print("\nRESULT: Basic polygon checks passed. Visual alignment still needs review.")

