import numpy as np
from s1.yolo_detector import drop_contained_boxes

def row(x1, y1, x2, y2, c=0.9, k=2):
    return [x1, y1, x2, y2, c, k]

def test_bumper_inside_car_dropped():
    d = np.array([row(0, 0, 400, 200), row(10, 120, 90, 190)], np.float32)
    out = drop_contained_boxes(d)
    assert len(out) == 1 and out[0, 2] == 400

def test_motorcycle_in_front_of_car_kept():
    d = np.array([row(0, 0, 400, 200, k=2), row(10, 120, 90, 190, k=3)], np.float32)
    assert len(drop_contained_boxes(d)) == 2

def test_touching_same_class_kept():
    d = np.array([row(0, 0, 100, 100), row(100, 0, 200, 100)], np.float32)
    assert len(drop_contained_boxes(d)) == 2

def test_empty_and_single():
    assert len(drop_contained_boxes(np.zeros((0, 6), np.float32))) == 0
    assert len(drop_contained_boxes(np.array([row(0, 0, 10, 10)], np.float32))) == 1
