"""Reference-frame camera motion estimation for virtual parking geometry.

The returned affine matrix maps points in the CURRENT source frame into the
REFERENCE source frame. Detection and tracking remain in current-frame pixels.
"""
from __future__ import annotations

import csv
import math
import os
import time

import cv2
import numpy as np


def transform_points(points, matrix):
    """Apply a 2x3 affine transform to an iterable of (x, y) points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.transform(pts, np.asarray(matrix, dtype=np.float32)).reshape(-1, 2)


def transform_bbox(bbox, matrix):
    """Transform a bbox's four corners and return its reference-space AABB."""
    x1, y1, x2, y2 = map(float, bbox)
    pts = transform_points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], matrix)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


class CameraMotionEstimator:
    """Estimate current-to-reference motion with ORB/RANSAC and ECC fallback.

    Low-quality estimates are rejected and the last good transform is held.
    Diagnostics are appended to CSV when ``diagnostics_path`` is supplied.
    """
    def __init__(self, diagnostics_path=None, max_dimension=800, min_inliers=8,
                 max_translation_fraction=0.10, max_rotation_deg=8.0,
                 min_scale=0.90, max_scale=1.10, smoothing=0.25):
        self.max_dimension = int(max_dimension)
        self.min_inliers = int(min_inliers)
        self.max_translation_fraction = float(max_translation_fraction)
        self.max_rotation_deg = float(max_rotation_deg)
        self.min_scale, self.max_scale = float(min_scale), float(max_scale)
        self.smoothing = float(smoothing)
        self.orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference_gray = None
        self.reference_size = None
        self.reference_kp = None
        self.reference_desc = None
        self.last_good = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        self.quality = "reference"
        self.inliers = 0
        self._csv_path = diagnostics_path
        self._csv_ready = False

    def _gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        scale = min(1.0, float(self.max_dimension) / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=cv2.INTER_AREA)
        return gray, scale

    def _log(self, frame_id, matrix, inliers, method, accepted):
        if not self._csv_path:
            return
        parent = os.path.dirname(self._csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        new_file = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        a = np.asarray(matrix, dtype=float)
        scale = math.sqrt(max(0.0, a[0, 0] ** 2 + a[1, 0] ** 2))
        rotation = math.degrees(math.atan2(a[1, 0], a[0, 0]))
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "frame_id", "method", "accepted", "inliers",
                                 "tx_px", "ty_px", "rotation_deg", "scale", "quality"])
            writer.writerow([f"{time.time():.3f}", frame_id, method, int(accepted), int(inliers),
                             f"{a[0, 2]:.3f}", f"{a[1, 2]:.3f}", f"{rotation:.4f}",
                             f"{scale:.6f}", self.quality])
    def _log(
        self,
        frame_id,
        matrix,
        inliers,
        method,
        accepted,
        rejection_reason="",
        ecc_score=None,
        candidate_matrix=None,
    ):
        if not self._csv_path:
            return

        parent = os.path.dirname(self._csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        new_file = (
            not os.path.exists(self._csv_path)
            or os.path.getsize(self._csv_path) == 0
        )

        # Log the candidate separately from the last accepted transform.
        candidate = (
            np.asarray(candidate_matrix, dtype=float)
            if candidate_matrix is not None
            else None
        )
        held = np.asarray(matrix, dtype=float)

        def metrics(a):
            if a is None or a.shape != (2, 3) or not np.isfinite(a).all():
                return ["", "", "", ""]
            scale = math.hypot(a[0, 0], a[1, 0])
            rotation = math.degrees(
                math.atan2(a[1, 0], a[0, 0])
            )
            return [
                f"{a[0, 2]:.3f}",
                f"{a[1, 2]:.3f}",
                f"{rotation:.4f}",
                f"{scale:.6f}",
            ]

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow([
                    "time", "frame_id", "method", "accepted", "inliers",
                    "candidate_tx_px", "candidate_ty_px",
                    "candidate_rotation_deg", "candidate_scale",
                    "rejection_reason", "ecc_score",
                    "held_tx_px", "held_ty_px",
                    "held_rotation_deg", "held_scale", "quality",
                ])

            writer.writerow([
                f"{time.time():.3f}",
                frame_id,
                method,
                int(accepted),
                int(inliers),
                *metrics(candidate),
                rejection_reason,
                "" if ecc_score is None else f"{ecc_score:.5f}",
                *metrics(held),
                self.quality,
            ])


    def _to_source_matrix(self, small_matrix, scale):
        # S^-1 * A_small * S converts a transform between resized images to
        # the corresponding transform in original source-frame coordinates.
        s_inv = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]])
        s = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        return (s_inv @ np.vstack([small_matrix, [0, 0, 1]]) @ s)[:2].astype(np.float32)

    
    def _rejection_reason(self, matrix, inliers, gray_shape):
        """Explain why a candidate transform is rejected."""
        if matrix is None:
            return "no_candidate"

        if inliers < self.min_inliers:
            return "insufficient_inliers"

        a = np.asarray(matrix, dtype=float)
        if a.shape != (2, 3) or not np.isfinite(a).all():
            return "invalid_matrix"

        scale = math.hypot(a[0, 0], a[1, 0])
        rotation = abs(math.degrees(math.atan2(a[1, 0], a[0, 0])))

        if self.reference_size:
            w, h = self.reference_size
        else:
            h, w = gray_shape

        translation = math.hypot(a[0, 2], a[1, 2])

        if scale < self.min_scale or scale > self.max_scale:
            return "scale_out_of_range"

        if rotation > self.max_rotation_deg:
            return "rotation_out_of_range"

        if translation > self.max_translation_fraction * math.hypot(w, h):
            return "translation_out_of_range"

        return "accepted"

    def _acceptable(self, matrix, inliers, gray_shape):
        return self._rejection_reason(matrix, inliers, gray_shape) == "accepted"


    def update(self, image, moving_boxes=(), frame_id=None):
        gray, scale = self._gray(image)
        h, w = gray.shape[:2]
        if self.reference_gray is None:
            self.reference_gray = gray
            self.reference_size = image.shape[1], image.shape[0]
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            self.reference_kp, self.reference_desc = self.orb.detectAndCompute(gray, mask)
            self.quality, self.inliers = "reference", 0
            self._log(frame_id, self.last_good, 0, "reference", True)
            return self.last_good.copy()

        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for box in moving_boxes or ():
            x1, y1, x2, y2 = map(float, box)
            x1, x2 = int(max(0, x1 * scale)), int(min(w, x2 * scale))
            y1, y2 = int(max(0, y1 * scale)), int(min(h, y2 * scale))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        kp, desc = self.orb.detectAndCompute(gray, mask)
        candidate, inliers, method = None, 0, "orb_ransac"
        if desc is not None and self.reference_desc is not None and len(desc) >= 8 and len(self.reference_desc) >= 8:
            pairs = self.matcher.knnMatch(desc, self.reference_desc, k=2)
            good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
            if len(good) >= self.min_inliers:
                src = np.float32([kp[m.queryIdx].pt for m in good])
                dst = np.float32([self.reference_kp[m.trainIdx].pt for m in good])
                small, inlier_mask = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                    maxIters=2000, confidence=0.99, refineIters=10)
                inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
                if small is not None:
                    candidate = self._to_source_matrix(small, scale)

        orb_candidate = candidate
        orb_inliers = inliers
        rejection_reason = self._rejection_reason(
            candidate, inliers, gray.shape
        )
        ecc_score = None

        accepted = rejection_reason == "accepted"
        if not accepted and desc is not None and self.reference_desc is not None and len(desc) >= 8 and len(self.reference_desc) >= 8:
            # ECC is only a fallback for small, near-identity changes. Skip it
            # entirely on blank/textureless frames (where it cannot converge).
            # OpenCV's returned warp is reference->current for inverse-map
            # alignment; invert it to preserve current->reference convention.
            method = "ecc_fallback"
            try:
                warp = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-5)
                score, warp = cv2.findTransformECC(self.reference_gray, gray, warp,
                                                   cv2.MOTION_AFFINE, criteria, None, 1)
                ecc_score = float(score)
                inv = cv2.invertAffineTransform(warp)
                candidate = self._to_source_matrix(inv, scale)
                inliers = 0
                rejection_reason = self._rejection_reason(
                    candidate, self.min_inliers, gray.shape
                )
                if score < 0.75:
                    rejection_reason = "ecc_score_too_low"
                accepted = rejection_reason == "accepted"
            except cv2.error:
                candidate, accepted = None, False
                rejection_reason = "ecc_error"

        if accepted:
            # Smooth the affine parameters rather than blindly averaging matrix
            # coefficients, which can introduce shear.
            old = self.last_good.astype(float)
            new = candidate.astype(float)
            old_scale = math.sqrt(old[0, 0] ** 2 + old[1, 0] ** 2)
            new_scale = math.sqrt(new[0, 0] ** 2 + new[1, 0] ** 2)
            old_angle = math.atan2(old[1, 0], old[0, 0])
            new_angle = math.atan2(new[1, 0], new[0, 0])
            alpha = self.smoothing
            sc = (1 - alpha) * old_scale + alpha * new_scale
            angle = old_angle + alpha * math.atan2(math.sin(new_angle-old_angle), math.cos(new_angle-old_angle))
            tx = (1-alpha) * old[0, 2] + alpha * new[0, 2]
            ty = (1-alpha) * old[1, 2] + alpha * new[1, 2]
            self.last_good = np.array([[sc*math.cos(angle), -sc*math.sin(angle), tx],
                                       [sc*math.sin(angle), sc*math.cos(angle), ty]], dtype=np.float32)
            self.quality = "good" if method == "orb_ransac" else "ecc"
        else:
            self.quality = "held_last_good"
        self.inliers = inliers
        self._log(
            frame_id,
            self.last_good,
            inliers,
            method,
            accepted,
            rejection_reason=rejection_reason,
            ecc_score=ecc_score,
            candidate_matrix=candidate,
        )
        return self.last_good.copy()