import json
import math


class Geometry:
    """Tripwire, slots and plate zone, all in source-frame pixel coordinates."""

    def __init__(self, data):
        self.frame_size = tuple(data.get("frame_size") or (0, 0))
        self.tripwire = [tuple(p) for p in data["tripwire"]]
        self.inside_point = tuple(data["inside_point"])
        self.plate_zone = data.get("plate_zone")
        self.slots = data.get("slots", [])

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def scaled(self, w, h):
        """Rescale if the stream resolution differs from the calibration resolution."""
        rw, rh = self.frame_size
        if not rw or not rh or (rw, rh) == (w, h):
            return self
        sx, sy = w / rw, h / rh
        sp = lambda p: (p[0] * sx, p[1] * sy)
        return Geometry({
            "frame_size": (w, h),
            "tripwire": [sp(p) for p in self.tripwire],
            "inside_point": sp(self.inside_point),
            "plate_zone": [sp(p) for p in self.plate_zone] if self.plate_zone else None,
            "slots": [{**s, "polygon": [sp(p) for p in s["polygon"]]} for s in self.slots],
        })


def signed_distance(p1, p2, x, y):
    """Signed perpendicular distance (px) of (x, y) from the line p1->p2."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length


def assign_ground_point(point, slots, previous_slot=None, boundary_margin_px=6.0, occupied_slots=None):
    """Assign a ground-contact point to a polygon with boundary hysteresis.

    ``slots`` is an iterable of mappings containing ``slot_id`` and ``polygon``.
    Occupied slots are excluded unless they are the track's previous slot.
    Returns a slot ID or ``None`` when the point is ambiguous/outside.
    """
    x, y = float(point[0]), float(point[1])
    occupied = set(occupied_slots or ())
    candidates = []
    for slot in slots:
        sid = slot["slot_id"]
        if sid in occupied and sid != previous_slot:
            continue
        poly = __import__("numpy").asarray(slot["polygon"], dtype="float32").reshape(-1, 1, 2)
        signed = float(__import__("cv2").pointPolygonTest(poly, (x, y), True))
        if signed >= 0:
            candidates.append((sid, signed))
        elif sid == previous_slot and signed >= -float(boundary_margin_px):
            # Keep the existing bay through small ground-point jitter near its edge.
            return sid
    if not candidates:
        return None
    # If polygons overlap, prefer the deepest interior point; stable slot order
    # breaks exact ties deterministically.
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]