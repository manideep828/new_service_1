import os
import tempfile

from s1.s2_mapper import S2Mapper


class FakePublisher:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


def make_event(event_type, timestamp):
    plate = "TG15BA2444"
    return {
        "event_id": f"test-{event_type}-{timestamp}",
        "event_type": event_type,
        "camera_id": "CAM-01",
        "parking_area_id": "AREA-01",
        "timestamp": timestamp,
        "event_time": timestamp,
        "captured_at": None,
        "received_at": None,
        "plate_status": "INHERITED",
        "plate_donor_uid": "donor-1",
        "vehicle": {"plate": plate},
        "vehicle_details": {"plate": plate},
        "plate": {"text": plate},
        "parking": {"slot_id": "B-01"},
    }


def test_inherited_plate_parking_lifecycle_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        inner = FakePublisher()
        db_path = os.path.join(tmp, "lifecycle.db")
        mapper = S2Mapper(inner, db_path=db_path)

        try:
            mapper.publish(make_event(
                "PARK_START", "2026-10-21T10:00:00+05:30"
            ))
            assert len(inner.events) == 1
            entry = inner.events[0]
            assert entry["event_type"] == "ENTRY"
            assert entry["vehicle"]["plate"] == "TG15BA2444"
            assert entry["plate_status"] == "INHERITED"

            mapper.publish(make_event(
                "PARK_END", "2026-10-21T11:00:00+05:30"
            ))
            assert len(inner.events) == 2
            exit_event = inner.events[1]
            assert exit_event["event_type"] == "EXIT"
            assert exit_event["vehicle"]["plate"] == "TG15BA2444"
            assert exit_event["plate_status"] == "INHERITED"
            assert exit_event["plate_donor_uid"] == "donor-1"
            assert exit_event["received_at"] == "2026-10-21T11:00:00+05:30"
        finally:
            mapper._db.close()
