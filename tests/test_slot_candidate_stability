import unittest

import numpy as np

from s1.logic import SlotManager
from s1.vehicles import Vehicle


class Track:
    def __init__(self, track_id, bbox, history):
        self.id = track_id
        self.bbox = bbox
        self.history = history


class Registry:
    def __init__(self, track_id=1):
        vehicle = Vehicle(uid=f"vehicle-{track_id}", track_id=track_id, first_ts=0.0)
        self.by_track = {track_id: vehicle}
        self.by_uid = {vehicle.uid: vehicle}

    def rebind(self, track_id, vehicle):
        self.by_track[track_id] = vehicle


def manager():
    slots = [
        {"slot_id": "A", "level": "L1", "polygon": [[0, 10], [90, 10], [90, 90], [0, 90]]},
        {"slot_id": "B", "level": "L1", "polygon": [[100, 10], [190, 10], [190, 90], [100, 90]]},
    ]
    cfg = {
        "footprint_frac": 0.25,
        "min_overlap": 0.3,
        "leave_overlap": 0.3,
        "park_dwell_s": 4.0,
        "leave_dwell_s": 3.0,
        "still_frac": 0.12,
        "moved_frac": 0.35,
        "lost_grace_s": 30.0,
        "baseline_window_s": 0.0,
        "candidate_gap_grace_s": 0.6,
        "candidate_switch_confirm_s": 0.6,
    }
    return SlotManager(slots, (200, 100), cfg)


def track(track_id, bbox, ts, cx, cy, width=40):
    # Stationary history lets the tests isolate slot-candidate behavior.
    hist = [(t, cx, cy, width) for t in (0.0, max(0.0, ts - 0.2), ts)]
    return Track(track_id, bbox, hist)


class TestSlotCandidateStability(unittest.TestCase):
    def test_brief_unassigned_frame_does_not_reset_candidate_dwell(self):
        slots = manager()
        registry = Registry()
        slots.update([track(1, (20, 20, 60, 70), 0.0, 40, 68)], 0.0, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")
        self.assertEqual(slots.tstate[1]["candidate_since"], 0.0)

        # A transient bad detection falls outside all calibrated bays.
        slots.update([track(1, (300, 20, 340, 70), 0.2, 40, 68)], 0.2, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")
        self.assertEqual(slots.tstate[1]["candidate_since"], 0.0)

        slots.update([track(1, (20, 20, 60, 70), 0.4, 40, 68)], 0.4, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")
        self.assertEqual(slots.tstate[1]["candidate_since"], 0.0)

    def test_new_slot_must_persist_before_candidate_switch(self):
        slots = manager()
        registry = Registry()
        slots.update([track(1, (20, 20, 60, 70), 0.0, 40, 68)], 0.0, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")

        slots.update([track(1, (120, 20, 160, 70), 0.2, 140, 68)], 0.2, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")
        self.assertEqual(slots.tstate[1]["pending_slot"], "B")

        slots.update([track(1, (120, 20, 160, 70), 0.5, 140, 68)], 0.5, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "A")

        slots.update([track(1, (120, 20, 160, 70), 0.9, 140, 68)], 0.9, registry)
        self.assertEqual(slots.tstate[1]["candidate_slot"], "B")
        self.assertEqual(slots.tstate[1]["candidate_since"], 0.9)
        self.assertIsNone(slots.tstate[1]["pending_slot"])

    def test_gap_longer_than_grace_clears_candidate(self):
        slots = manager()
        registry = Registry()
        slots.update([track(1, (20, 20, 60, 70), 0.0, 40, 68)], 0.0, registry)
        slots.update([track(1, (300, 20, 340, 70), 0.7, 40, 68)], 0.7, registry)
        self.assertIsNone(slots.tstate[1]["candidate_slot"])
        self.assertIsNone(slots.tstate[1]["candidate_since"])


if __name__ == "__main__":
    unittest.main()
