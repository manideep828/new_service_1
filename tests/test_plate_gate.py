from types import SimpleNamespace as NS
from s1.events import build_event

class FakeVehicle:
    def __init__(self, final):
        self.voter = NS(consensus=lambda: NS(text="TG15A2444", conf=0.777))
        self.plate_final = final
        self.uid = "u1"
        self.track_id = 1
        self.first_ts = 0.0
        self.slot_id = None
        self.emitted_plate = None

    def vehicle_type(self):
        return "car"

    def color(self):
        return "black"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None

def test_unconfirmed_gives_null():
    ev = build_event("PARK_START", FakeVehicle(False), "CAM-01", 1.0, 1.0)
    assert ev["plate"]["text"] is None
    assert ev["vehicle"]["plate"] is None

def test_confirmed_gives_plate():
    ev = build_event("PARK_START", FakeVehicle(True), "CAM-01", 1.0, 1.0)
    assert ev["plate"]["text"] == "TG15A2444"

def test_inherited_plate_does_not_fake_ocr_confirmation():
    v = FakeVehicle(False)
    v.inherited_plate = "TG15A2444"
    v.plate_donor_uid = "donor-123"

    ev = build_event("PARK_START", v, "CAM-01", 1.0, 1.0)

    # The base event builder must not pretend inherited OCR is confirmed.
    assert ev["plate"]["text"] is None
    assert ev["vehicle"]["plate"] is None
    assert v.plate_final is False
