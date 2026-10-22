from types import SimpleNamespace as NS

from s1.config import load_config
from s1.geometry import Geometry
from s1.pipeline import Pipeline
from s1.vehicles import Vehicle


W, H = 640, 480
GEO = {
    "frame_size": [W, H],
    "tripwire": [[0, 100], [640, 100]],
    "inside_point": [320, 300],
    "plate_zone": None,
    "slots": [
        {
            "slot_id": "B-01",
            "level": "L1",
            "polygon": [[300, 240], [460, 240], [460, 340], [300, 340]],
        }
    ],
}


class FakeDetector:
    def detect(self, image):
        return []


class FakePublisher:
    def publish(self, event):
        pass


def test_inherited_plate_fields_are_consistent():
    cfg = load_config(None, {
        "parking_area_id": "AREA-01",
        "publisher": {"snapshot_dir": None},
    })

    pipe = Pipeline(
        cfg,
        Geometry(GEO),
        (W, H),
        FakeDetector(),
        FakePublisher(),
        plate_worker=None,
        now_fn=lambda: 1800000000.0,
    )

    v = Vehicle(uid="recipient-1", track_id=7, first_ts=1800000000.0)
    v.cls_votes[2] = 3
    v.inherited_plate = "TG15BA2444"
    v.plate_donor_uid = "donor-1"
    assert v.plate_final is False

    pending = {
        "event_id": "test-inherited-event",
        "event_type": "PARK_START",
        "vehicle": v,
        "event_time": 1800000001.0,
        "detected_at": 1800000002.0,
        "direction": "entry",
        "slot": {"slot_id": "B-01", "level": "L1"},
        "inferred": False,
    }

    out = []
    pipe._build_and_append(out, pending)

    assert len(out) == 1
    ev = out[0]
    assert ev["vehicle"]["plate"] == "TG15BA2444"
    assert ev["vehicle_details"]["plate"] == "TG15BA2444"
    assert ev["plate"]["text"] == "TG15BA2444"
    assert ev["plate_status"] == "INHERITED"
    assert ev["plate_donor_uid"] == "donor-1"
    assert v.plate_final is False
