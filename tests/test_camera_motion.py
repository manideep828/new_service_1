import unittest

import cv2
import numpy as np

from s1.camera_motion import CameraMotionEstimator, transform_points
from s1.geometry import assign_ground_point


class TestGroundPointAssignment(unittest.TestCase):
    def setUp(self):
        self.slots = [
            {"slot_id": "B-01", "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"slot_id": "B-02", "polygon": [(100, 0), (200, 0), (200, 100), (100, 100)]},
        ]

    def test_point_inside_slot(self):
        self.assertEqual(assign_ground_point((40, 50), self.slots), "B-01")
        self.assertEqual(assign_ground_point((160, 50), self.slots), "B-02")

    def test_boundary_hysteresis_keeps_previous_slot(self):
        self.assertEqual(
            assign_ground_point((103, 50), self.slots, previous_slot="B-01", boundary_margin_px=5),
            "B-01",
        )

    def test_occupied_slot_cannot_be_claimed_by_another_track(self):
        self.assertIsNone(
            assign_ground_point((40, 50), self.slots, occupied_slots={"B-01"})
        )
        self.assertEqual(
            assign_ground_point((40, 50), self.slots, previous_slot="B-01", occupied_slots={"B-01"}),
            "B-01",
        )

    def test_overlap_tie_uses_deepest_interior(self):
        overlap = [
            {"slot_id": "A", "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {"slot_id": "B", "polygon": [(20, 0), (120, 0), (120, 100), (20, 100)]},
        ]
        self.assertEqual(assign_ground_point((75, 50), overlap), "B")


class TestCameraMotion(unittest.TestCase):
    @staticmethod
    def textured_frame(seed=7, size=(480, 320)):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def test_transform_maps_current_to_reference(self):
        reference = self.textured_frame()
        motion = np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float32)
        current = cv2.warpAffine(reference, motion, (reference.shape[1], reference.shape[0]))
        estimator = CameraMotionEstimator(diagnostics_path=None, min_inliers=8, smoothing=1.0)
        estimator.update(reference, frame_id=0)
        matrix = estimator.update(current, frame_id=1)
        mapped = transform_points([(150, 120)], matrix)[0]
        # Current image content was shifted +5,+3, so its point maps back -5,-3.
        self.assertLess(abs(float(mapped[0]) - 145), 3.0)
        self.assertLess(abs(float(mapped[1]) - 117), 3.0)
        self.assertIn(estimator.quality, {"good", "ecc"})

    def test_stationary_ground_point_stays_in_same_reference_slot_after_camera_shift(self):
        reference_point = (150.0, 120.0)
        # Simulate the camera moving content +12,+7 in the current frame.
        current_point = (reference_point[0] + 12, reference_point[1] + 7)
        current_to_reference = np.array([[1, 0, -12], [0, 1, -7]], dtype=np.float32)
        mapped = transform_points([current_point], current_to_reference)[0]
        slots = [{"slot_id": "B-01", "polygon": [(100, 80), (200, 80), (200, 180), (100, 180)]}]
        self.assertEqual(assign_ground_point(reference_point, slots), "B-01")
        self.assertEqual(assign_ground_point(mapped, slots), "B-01")

    def test_poor_frame_holds_last_good_transform(self):
        reference = self.textured_frame()
        estimator = CameraMotionEstimator(diagnostics_path=None, min_inliers=8)
        first = estimator.update(reference, frame_id=0)
        before = estimator.update(cv2.warpAffine(reference, np.float32([[1, 0, 4], [0, 1, 2]]),
                                                  (reference.shape[1], reference.shape[0])), frame_id=1)
        blank = np.zeros_like(reference)
        after = estimator.update(blank, frame_id=2)
        self.assertTrue(np.allclose(after, before))
        self.assertEqual(estimator.quality, "held_last_good")
        self.assertEqual(first.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
