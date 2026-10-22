from types import SimpleNamespace as NS
from s1.plate_inherit import find_plate_donor, edit_distance

def iou(a, b):
    return 0.33 if (a, b) == ("rb", "db") else 0.0

def donor(**kw):
    d = dict(plate_final=True, slot_id=None, track_id=1, first_ts=0.0, last_seen_ts=6.0,
             last_bbox="db", voter=NS(consensus=lambda: NS(text="TG15BA2444")))
    d.update(kw)
    return NS(**d)

def recip():
    return NS(first_ts=4.0, first_bbox="rb")

def run(donors, by_track=None, cand="TG15A2444"):
    return find_plate_donor(recip(), cand, donors, by_track or {}, iou)

def test_edit_distance():
    assert edit_distance("TG15A2444", "TG15BA2444") == 1

def test_single_donor_inherits():
    d = donor()
    assert run([d]) is d

def test_two_donors_inherit_nothing():
    assert run([donor(), donor(track_id=2)]) is None

def test_alive_track_rejected():
    d = donor()
    assert run([d], {1: d}) is None

def test_slot_rejected():
    assert run([donor(slot_id="B-01")]) is None

def test_no_time_overlap_rejected():
    assert run([donor(last_seen_ts=3.0)]) is None

def test_too_many_edits_rejected():
    assert run([donor()], cand="AB12CD3456") is None

def test_low_iou_rejected():
    d = donor(last_bbox="other")
    assert run([d]) is None
