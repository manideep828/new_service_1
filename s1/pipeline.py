import base64
import logging
import os
import time
import uuid
from collections import Counter, deque

import cv2
import numpy as np

from .attributes import dominant_color
from .events import build_event
from .logic import SlotManager, Tripwire, still_run_start
from .camera_motion import CameraMotionEstimator, transform_points
from .plates import PlateVoter, crop_quality
from .plate_crop import cut_plate
from .plates import fix_indian
from .tracker import ByteTracker
from .vehicles import Vehicle, VehicleRegistry

log = logging.getLogger("pipeline")

LIFECYCLE_EVENT_TYPES = frozenset({
    "ENTRY",
    "PARK_START",
    "PARK_END",
    "EXIT",
})


class Stats:
    def __init__(self, n=200):
        self.d = {k: deque(maxlen=n) for k in ("detect", "track", "total", "e2e")}
        self.frames = 0
        self.t0 = time.time()

    def add(self, k, v):
        self.d[k].append(v)

    def summary(self):
        def p(k, q):
            a = self.d[k]
            return float(np.percentile(a, q)) if a else 0.0

        fps = self.frames / max(time.time() - self.t0, 1e-6)
        return (
            f"fps={fps:4.1f} detect={p('detect', 50):5.0f}ms track={p('track', 50):4.1f}ms "
            f"total p50={p('total', 50):5.0f} p95={p('total', 95):5.0f}ms "
            f"capture->event-ready p50={p('e2e', 50):5.0f} p95={p('e2e', 95):5.0f}ms"
        )


class Pipeline:
    def __init__(
        self,
        cfg,
        geometry,
        frame_size,
        detector,
        publisher,
        plate_worker=None,
        now_fn=time.time,
    ):
        self.cfg, self.cam = cfg, cfg["camera_id"]
        self.now = now_fn
        self.detector, self.publisher, self.plates = detector, publisher, plate_worker

        t = cfg["tracker"]
        self.tracker = ByteTracker(
            t["high_thr"],
            t["low_thr"],
            t["match_iou"],
            t["low_match_iou"],
            t["max_age_s"],
            t["min_hits"],
        )

        g = geometry.scaled(*frame_size)
        self.geo = g
        motion_cfg = cfg.get("camera_motion", {})
        motion_cfg = cfg.get("camera_motion", {})
        self.camera_motion_enabled = motion_cfg.get("enabled", True)

        self.camera_motion = CameraMotionEstimator(
        diagnostics_path=motion_cfg.get(
            "diagnostics_path", "data/camera_motion_diagnostics.csv"
        ),
        max_dimension=motion_cfg.get("max_dimension", 800),
        min_inliers=motion_cfg.get("min_inliers", 8),
        )	
        self.camera_transform = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)

        self.tripwire = Tripwire(
            g.tripwire[0],
            g.tripwire[1],
            g.inside_point,
            cfg["gate"]["margin_px"],
            size_samples=cfg["gate"]["size_samples"],
            min_size_change=cfg["gate"]["min_size_change"],
        )

        self.slots = SlotManager(g.slots, frame_size, cfg["parking"])

        self.zone = (
            np.array(g.plate_zone, np.int32).reshape(-1, 1, 2)
            if g.plate_zone
            else None
        )

        self.registry = VehicleRegistry()
        self.stats = Stats()
        self.snap_dir = cfg["publisher"]["snapshot_dir"]
        self.last_tracks = []

        # All four lifecycle events remain pending until a finalized,
        # valid plate is available.
        self.pending_parking_events = []

        # Maps stable event IDs to their durable pending-event keys.
        self._durable_pending_keys = {}

        self._restore_pending_parking_events()

    # ------------------------------------------------------------------
    # Durable pending-event serialization and recovery
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_crop(crop):
        if crop is None:
            return None

        try:
            ok, encoded = cv2.imencode(
                ".jpg",
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            if not ok:
                return None
            return base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception:
            log.exception("Could not encode crop for pending-event storage")
            return None

    @staticmethod
    def _decode_crop(encoded):
        if not encoded:
            return None

        try:
            raw = base64.b64decode(encoded, validate=True)
            array = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception:
            log.exception("Could not restore crop for pending parking event")
            return None

    def _serialize_pending(self, pending):
        v = pending["vehicle"]

        return {
            "event_id": pending["event_id"],
            "event_type": pending["event_type"],
            "event_time": pending["event_time"],
            "detected_at": pending["detected_at"],
            "received_at": pending.get("received_at"),
            "captured_at": pending.get("captured_at"),
            "direction": pending["direction"],
            "slot": pending["slot"],
            "inferred": pending["inferred"],
            "created_at": pending["created_at"],
            "deadline": pending["deadline"],
            "timed_out": pending["timed_out"],
            "recovery_crop_retry": pending.get("recovery_crop_retry", False),
            "recovery_retry_due": pending.get("recovery_retry_due", False),
            "vehicle": {
                "uid": v.uid,
                "track_id": v.track_id,
                "first_ts": v.first_ts,
                "cls_votes": list(v.cls_votes.items()),
                "color_votes": list(v.color_votes.items()),
                "plate_reads": [list(read) for read in v.voter.reads],
                "plate_max_reads": v.voter.max_reads,
                "plate_final": v.plate_final,
                "plate_attempts": v.plate_attempts,
                "plate_inflight": v.plate_inflight,
                "last_plate_submit": v.last_plate_submit,
                "last_color_ts": v.last_color_ts,
                "last_crop": self._encode_crop(v.last_crop),
                "any_emitted": v.any_emitted,
                "emitted_plate": v.emitted_plate,
                "slot_id": v.slot_id,
            },
        }

    def _deserialize_pending(self, payload):
        if payload.get("event_type") not in LIFECYCLE_EVENT_TYPES:
            raise ValueError("Unsupported pending lifecycle event type")

        if not payload.get("event_id"):
            raise ValueError("Pending event has no stable event_id")

        vehicle_data = payload["vehicle"]

        v = Vehicle(
            uid=vehicle_data["uid"],
            track_id=int(vehicle_data["track_id"]),
            first_ts=float(vehicle_data["first_ts"]),
        )

        v.cls_votes = Counter(
            {
                int(key): int(value)
                for key, value in vehicle_data.get("cls_votes", [])
            }
        )
        v.color_votes = Counter(
            {
                str(key): int(value)
                for key, value in vehicle_data.get("color_votes", [])
            }
        )

        voter = PlateVoter()
        voter.max_reads = int(
            vehicle_data.get("plate_max_reads", voter.max_reads)
        )
        voter.reads = [
            (str(read[0]), float(read[1]), bool(read[2]))
            for read in vehicle_data.get("plate_reads", [])
        ]
        v.voter = voter

        v.plate_final = bool(vehicle_data.get("plate_final", False))
        v.plate_attempts = int(vehicle_data.get("plate_attempts", 0))
        v.last_plate_submit = float(
            vehicle_data.get("last_plate_submit", 0.0)
        )
        v.last_color_ts = float(vehicle_data.get("last_color_ts", 0.0))

        saved_inflight = bool(vehicle_data.get("plate_inflight", False))

        # OCR jobs from the previous process no longer exist.
        v.plate_inflight = False
        v.last_crop = self._decode_crop(vehicle_data.get("last_crop"))

        # An interrupted OCR job consumed an attempt but never delivered its
        # result. Credit that attempt back so the existing retry limit remains
        # useful after a restart.
        if saved_inflight and not v.plate_final:
            v.plate_attempts = max(0, v.plate_attempts - 1)

        can_retry_crop = bool(v.last_crop is not None and not v.plate_final)

        v.any_emitted = bool(vehicle_data.get("any_emitted", False))
        v.emitted_plate = vehicle_data.get("emitted_plate")
        v.slot_id = vehicle_data.get("slot_id")

        return {
            "event_id": payload["event_id"],
            "event_type": payload["event_type"],
            "vehicle": v,
            "event_time": float(payload["event_time"]),
            "detected_at": float(payload["detected_at"]),
            "direction": payload.get("direction"),
            "slot": payload.get("slot"),
            "inferred": bool(payload.get("inferred", False)),
            "created_at": float(payload["created_at"]),
            "deadline": float(payload["deadline"]),
            "timed_out": bool(payload.get("timed_out", False)),
            "recovery_crop_retry": bool(
                can_retry_crop
                and (
                    saved_inflight
                    or payload.get("recovery_crop_retry", False)
                    or v.last_crop is not None
                )
            ),
            "recovery_retry_due": bool(can_retry_crop),
        }

    def _save_pending_event(self, pending):
        save = getattr(self.publisher, "save_pending_event", None)
        if not callable(save):
            log.error(
                "Publisher does not support durable pending-event storage; "
                "event=%s remains memory-only",
                pending.get("event_id"),
            )
            return False

        event_key = pending["event_id"]

        try:
            save(event_key, self._serialize_pending(pending))
            self._durable_pending_keys[event_key] = event_key
            return True
        except Exception:
            log.exception(
                "Could not persist pending lifecycle event id=%s",
                event_key,
            )
            return False

    def _persist_all_pending_events(self):
        for pending in self.pending_parking_events:
            self._save_pending_event(pending)

    def _restore_pending_parking_events(self):
        load = getattr(self.publisher, "load_pending_events", None)
        if not callable(load):
            log.warning(
                "Publisher has no pending-event recovery interface"
            )
            return

        try:
            saved_events = load()
        except Exception:
            log.exception("Could not load durable pending lifecycle events")
            return

        for event_key, payload in saved_events:
            try:
                pending = self._deserialize_pending(payload)

                if event_key != pending["event_id"]:
                    raise ValueError(
                        "Stored event key does not match event_id"
                    )

                if not pending["vehicle"].uid:
                    raise ValueError("Pending event has no vehicle UID")

                self.pending_parking_events.append(pending)
                self._durable_pending_keys[event_key] = event_key

                # Keep OCR results addressable while this event is pending.
                self.registry.by_uid[pending["vehicle"].uid] = pending["vehicle"]

                log.info(
                    "Restored pending %s event_id=%s vehicle=%s "
                    "original_event_time=%s crop_retry=%s",
                    pending["event_type"],
                    event_key,
                    pending["vehicle"].uid,
                    pending["event_time"],
                    pending["recovery_crop_retry"],
                )

            except Exception:
                # Leave invalid rows in SQLite for inspection/recovery.
                log.exception(
                    "Could not restore pending event key=%s; "
                    "the database row has been retained",
                    event_key,
                )

        log.info(
            "Pending-event recovery loaded %d event(s)",
            len(self.pending_parking_events),
        )

    def _publish_event(self, event):
        """Persist to the outgoing outbox before deleting its pending row."""
        event_id = event["event_id"]
        pending_key = self._durable_pending_keys.get(event_id)

        # Outbox.publish commits to SQLite before returning.
        self.publisher.publish(event)

        if pending_key is None:
            return

        delete = getattr(self.publisher, "delete_pending_event", None)
        if not callable(delete):
            log.error(
                "Outgoing event %s was queued, but pending storage "
                "cannot delete its row; it will be recovered again",
                event_id,
            )
            return

        try:
            delete(pending_key)
            self._durable_pending_keys.pop(event_id, None)
        except Exception:
            # The stable event ID makes a later outbox INSERT OR IGNORE safe.
            log.exception(
                "Outgoing event %s was queued, but pending-row deletion failed",
                event_id,
            )

    # ------------------------------------------------------------------
    # Snapshots and event construction
    # ------------------------------------------------------------------

    def _snapshot(self, v, etype, ts):
        if v.last_crop is None or not self.snap_dir:
            return None

        d = os.path.join(
            self.snap_dir,
            time.strftime("%Y%m%d", time.gmtime(ts)),
        )
        os.makedirs(d, exist_ok=True)

        path = os.path.join(
            d,
            f"{v.uid[:8]}_{etype}_{int(ts * 1000)}.jpg",
        )
        cv2.imwrite(path, v.last_crop)
        return path

    def _plate_snapshot(self, v, etype, ts):
        _c = v.voter.consensus() if getattr(v, "plate_final", False) else None
        _crops = getattr(v, "plate_crops", None) or {}
        crop = None
        if _c and _crops:
            if _c.text in _crops:
                crop = _crops[_c.text][1]
            else:
                crop = max(_crops.values(), key=lambda t: t[0])[1]
        if crop is None or not self.snap_dir:
            return None
        d = os.path.join(self.snap_dir, time.strftime("%Y%m%d", time.gmtime(ts)))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{v.uid[:8]}_{etype}_{int(ts * 1000)}_plate.jpg")
        cv2.imwrite(path, crop)
        return path

    def _plate_ready(self, v):
        """Require the configured finalized plate consensus before releasing lifecycle events."""
        pc = self.cfg["plate"]
        return v.voter.is_final(
            pc["confirm_reads"],
            pc["confirm_conf"],
        )

    def _build_and_append(self, out, pending):
        """Build an event while preserving its original time and stable ID."""
        v = pending["vehicle"]

        ev = build_event(
            pending["event_type"],
            v,
            self.cam,
            pending["event_time"],
            pending["detected_at"],
            pending["direction"],
            pending["slot"],
            self._snapshot(
                v,
                pending["event_type"],
                pending["event_time"],
            ),
            pending["inferred"],
            captured_at=pending.get("captured_at"),
            received_at=pending.get("received_at"),
            parking_area_id=self.cfg["parking_area_id"],
            plate_snapshot_uri=self._plate_snapshot(
                v,
                pending["event_type"],
                pending["event_time"],
            ),
        )

        # Reuse the durable ID if a process restarts before publication.
        ev["event_id"] = pending["event_id"]

        c = v.voter.consensus()
        cand = c
        c = c if getattr(v, "plate_final", False) else None
        v.emitted_plate = c.text if c else None
        ev["plate_status"] = "CONFIRMED" if c else "UNCONFIRMED"
        ev["plate_candidate"] = cand.text if (cand and not c) else None

        if not c and getattr(v, "inherited_plate", None) is None and cand is not None:
            try:
                from .plate_inherit import find_plate_donor
                reg = self.registry
                donor = find_plate_donor(
                    v, cand.text, list(reg.by_uid.values()), reg.by_track, reg._iou
                )
                if donor is not None:
                    v.inherited_plate = donor.voter.consensus().text
                    v.plate_donor_uid = donor.uid
                    log.info(
                        "PLATE_INHERITED recipient=%s donor=%s plate=%s",
                        v.uid, donor.uid, v.inherited_plate,
                    )
            except Exception:
                log.exception("PLATE_INHERIT failed")

        _ip = getattr(v, "inherited_plate", None)
        if _ip and not c:
            if isinstance(ev.get("vehicle"), dict):
                ev["vehicle"]["plate"] = _ip
            if isinstance(ev.get("vehicle_details"), dict):
                ev["vehicle_details"]["plate"] = _ip
            if isinstance(ev.get("plate"), dict):
                ev["plate"]["text"] = _ip

            ev["plate_status"] = "INHERITED"
            ev["plate_donor_uid"] = getattr(v, "plate_donor_uid", None)
            v.emitted_plate = _ip
        log.info("EVENT_PLATE type=%s status=%s plate=%s vehicle_plate=%s donor=%s", pending["event_type"], ev.get("plate_status"), (ev.get("plate") or {}).get("text"), (ev.get("vehicle") or {}).get("plate"), ev.get("plate_donor_uid"))
        v.any_emitted = True
        out.append(ev)

        log.info(
            "Released %s event_id=%s vehicle=%s original_event_time=%s plate=%s",
            pending["event_type"],
            pending["event_id"],
            v.uid,
            ev["timestamp"],
            ((ev.get("plate") or {}).get("text")),
        )

    def _emit(
        self,
        out,
        etype,
        v,
        event_time,
        direction=None,
        sid=None,
        inferred=False,
        received_at=None,
        captured_at=None,
    ):
        slot = self.slots.slot_info(sid) if sid else None
        detected_at = self.now()

        pending = {
            "event_id": uuid.uuid4().hex,
            "event_type": etype,
            "vehicle": v,
            "event_time": event_time,
            "detected_at": detected_at,
            "received_at": received_at,
            "captured_at": captured_at,
            "direction": direction,
            "slot": slot,
            "inferred": inferred,
            "created_at": detected_at,
            "deadline": (
                detected_at
                + max(
                    0.0,
                    float(self.cfg["plate"].get("emit_deadline_s", 3.0)),
                )
            ),
            "timed_out": False,
            "recovery_crop_retry": False,
            "recovery_retry_due": False,
        }

        if etype in LIFECYCLE_EVENT_TYPES:
            # Persist before waiting for OCR or attempting publication.
            self.pending_parking_events.append(pending)
            self._save_pending_event(pending)

            log.info(
                "Holding %s event_id=%s vehicle=%s original_event_time=%s "
                "plate_ready=%s deadline=%.3f",
                etype,
                pending["event_id"],
                v.uid,
                event_time,
                self._plate_ready(v),
                pending["deadline"],
            )
            return

        # Non-lifecycle events retain their existing immediate behavior.
        self._build_and_append(out, pending)

    def _in_zone(self, x, y):
        return (
            self.zone is None
            or cv2.pointPolygonTest(
                self.zone,
                (float(x), float(y)),
                False,
            ) >= 0
        )

    def _maybe_submit_plate(self, t, v, frame):
        pc = self.cfg["plate"]

        if (
            self.plates is None
            or not pc["enabled"]
            or v.plate_final
            or v.plate_inflight
        ):
            return

        if (
            v.plate_attempts >= pc["max_attempts"]
            or frame.ts - v.last_plate_submit < pc["interval_s"]
            or v.last_plate_frame_idx == frame.idx
        ):
            return

        x1, y1, x2, y2 = t.bbox

        if (
            (y2 - y1) < pc["min_vehicle_height_px"]
            or not self._in_zone((x1 + x2) / 2, y2)
        ):
            return

        H, W = frame.image.shape[:2]
        px, py = 0.05 * (x2 - x1), 0.05 * (y2 - y1)

        # Clamp every crop boundary to the image dimensions.
        xa = max(0, min(W, int(x1 - px)))
        ya = max(0, min(H, int(y1 - py)))
        xb = max(0, min(W, int(x2 + px)))
        yb = max(0, min(H, int(y2 + py)))
        if xb <= xa or yb <= ya:
            return

        crop = frame.image[ya:yb, xa:xb]
        accepted, quality = crop_quality(
            crop,
            min_brightness=float(pc.get("min_crop_brightness", 8.0)),
            max_brightness=float(pc.get("max_crop_brightness", 247.0)),
            min_contrast=float(pc.get("min_crop_contrast", 5.0)),
            min_laplacian_var=float(pc.get("min_crop_laplacian_var", 2.0)),
        )
        if not accepted:
            log.debug(
                "PLATE_CROP_REJECT uid=%s frame=%s reason=%s metrics=%s",
                v.uid, frame.idx, quality.get("reason"), quality,
            )
            return

        if crop.size and v.plate_attempts < 3:
            import os
            os.makedirs("data/debug_live_crops", exist_ok=True)
            debug_path = (
                f"data/debug_live_crops/{v.uid[:8]}_attempt{v.plate_attempts}.jpg"
            )
            cv2.imwrite(debug_path, crop)
        log.info("PLATE_CROP uid=%s shape=%s attempts=%s", v.uid, crop.shape, v.plate_attempts)

        if crop.size and self.plates.submit(v.uid, crop.copy()):
            # Save the exact crop immediately. If this OCR job is interrupted
            # by a restart, it can be resubmitted without the old track.
            # Keep the largest accepted crop for recovery retries.
            # Area is a proxy for available detail, not a guarantee of sharpness.
            if (
                v.last_crop is None
                or crop.shape[0] * crop.shape[1]
                > v.last_crop.shape[0] * v.last_crop.shape[1]
            ):
                v.last_crop = crop.copy()
            v.plate_inflight = True
            v.plate_inflight_observation_id = round(float(frame.ts), 6)
            v.last_plate_frame_idx = int(frame.idx)
            v.plate_attempts += 1
            v.last_plate_submit = frame.ts

    def _maybe_submit_recovered_crops(self, frame_ts):
        """Retry saved OCR crops for recovered events without live tracks."""
        pc = self.cfg["plate"]

        if self.plates is None or not pc["enabled"]:
            return

        live_uids = {
            v.uid for v in self.registry.by_track.values()
        }

        for pending in self.pending_parking_events:
            if not pending.get("recovery_crop_retry", False):
                continue

            v = pending["vehicle"]

            if (
                v.uid in live_uids
                or v.plate_final
                or v.plate_inflight
                or v.last_crop is None
                or v.plate_attempts >= pc["max_attempts"]
            ):
                continue

            due_now = pending.get("recovery_retry_due", False)
            if (
                not due_now
                and frame_ts - v.last_plate_submit < pc["interval_s"]
            ):
                continue

            try:
                crop = v.last_crop.copy()
                if crop.size and self.plates.submit(v.uid, crop):
                    v.plate_inflight = True
                    v.plate_attempts += 1
                    v.last_plate_submit = frame_ts
                    pending["recovery_retry_due"] = False

                    log.info(
                        "Resubmitted saved OCR crop for pending %s "
                        "event_id=%s vehicle=%s attempt=%d/%d",
                        pending["event_type"],
                        pending["event_id"],
                        v.uid,
                        v.plate_attempts,
                        pc["max_attempts"],
                    )
            except Exception:
                log.exception(
                    "Could not resubmit saved OCR crop for event_id=%s",
                    pending["event_id"],
                )

    def _apply_plate_results(self):
        pc = self.cfg["plate"]

        for uid, reads, crop in (self.plates.poll() if self.plates else []):
            v = self.registry.by_uid.get(uid)

            if v is None:
                for pending in self.pending_parking_events:
                    candidate = pending["vehicle"]
                    if candidate.uid == uid:
                        v = candidate
                        break

            if v is None:
                log.warning("Ignoring OCR result for unknown vehicle uid=%s", uid)
                continue

            observation_id = v.plate_inflight_observation_id
            v.plate_inflight_observation_id = None
            v.plate_inflight = False

            def acceptable_read(r):
                if r.conf < float(pc["min_ocr_conf"]):
                    return False
                if r.det_conf < float(pc.get("min_plate_detector_conf", 0.30)):
                    return False
                try:
                    x1, y1, x2, y2 = map(float, r.bbox)
                    width, height = x2 - x1, y2 - y1
                    if width < float(pc.get("min_plate_width_px", 24)):
                        return False
                    if height < float(pc.get("min_plate_height_px", 8)):
                        return False
                    ratio = width / max(height, 1e-6)
                    if not (
                        float(pc.get("min_plate_aspect_ratio", 1.5))
                        <= ratio
                        <= float(pc.get("max_plate_aspect_ratio", 7.0))
                    ):
                        return False
                except (TypeError, ValueError):
                    return False
                return True

            good = [r for r in reads if acceptable_read(r)]

            if good:
                selected = max(
                    good,
                    key=lambda r: float(r.conf) * float(r.det_conf),
                )

                v.voter.add(
                    selected.text,
                    selected.conf,
                    observation_id=observation_id,
                )

                if (
                    v.last_crop is None
                    or crop.shape[0] * crop.shape[1]
                    > v.last_crop.shape[0] * v.last_crop.shape[1]
                ):
                    v.last_crop = crop.copy()

                plate_crop = cut_plate(crop, selected.bbox)
                plate_text, plate_valid = fix_indian(selected.text)

                if plate_crop is not None and plate_valid:
                    store = getattr(v, "plate_crops", None)
                    if store is None:
                        store = v.plate_crops = {}

                    previous = store.get(plate_text)
                    if previous is None or selected.conf >= previous[0]:
                        store[plate_text] = (selected.conf, plate_crop)

                v.plate_final = v.voter.is_final(
                    pc["confirm_reads"],
                    pc["confirm_conf"],
                )

                consensus = v.voter.consensus()
                log.info(
                    "PLATE_STATE uid=%s state=%s selected=%s "
                    "ocr_conf=%.3f det_conf=%.3f consensus=%s reads=%d "
                    "attempt=%d",
                    uid,
                    v.plate_status,
                    selected.text,
                    float(selected.conf),
                    float(selected.det_conf),
                    consensus.text if consensus else None,
                    len(v.voter.reads),
                    v.plate_attempts,
                )

                if v.plate_final:
                    for pending in self.pending_parking_events:
                        if pending["vehicle"].uid == uid:
                            pending["recovery_crop_retry"] = False
                            pending["recovery_retry_due"] = False

    def _release_ready_parking_events(self, out, final=False):
        """Queue finalized lifecycle events; keep unresolved events durable."""
        now = self.now()
        remaining = []
        local_preview = getattr(self.publisher, 'local_preview', False)

        for pending in self.pending_parking_events:
            v = pending["vehicle"]

            if self._plate_ready(v):
                self._build_and_append(out, pending)
                continue

            if local_preview and (
                now >= pending["deadline"] or final
            ):
                log.warning(
                    "LOCAL_PREVIEW_PLATE_UNKNOWN event=%s event_id=%s vehicle=%s",
                    pending["event_type"],
                    pending["event_id"],
                    v.uid,
                )
                self._build_and_append(out, pending)
                continue

            if now >= pending["deadline"] and not pending["timed_out"]:
                pending["timed_out"] = True
                log.error(
                    "PLATE_WAIT_TIMEOUT event=%s event_id=%s vehicle=%s "
                    "original_event_time=%s; event retained and NOT sent "
                    "without a valid finalized plate",
                    pending["event_type"],
                    pending["event_id"],
                    v.uid,
                    pending["event_time"],
                )

            if final:
                log.error(
                    "UNRESOLVED_LIFECYCLE_EVENT event=%s event_id=%s vehicle=%s "
                    "original_event_time=%s plate_final=%s; durable record "
                    "retained and event NOT sent",
                    pending["event_type"],
                    pending["event_id"],
                    v.uid,
                    pending["event_time"],
                    v.plate_final,
                )

            remaining.append(pending)

        self.pending_parking_events = remaining
        self._persist_all_pending_events()

    def process(self, frame):
        t0 = time.perf_counter()
        events = []

        dets = self.detector.detect(frame.image)
        t1 = time.perf_counter()

        tracks, removed = self.tracker.update(dets, frame.ts)
        t2 = time.perf_counter()

        # Estimate current->reference motion using current-frame track boxes as
        # masks. This keeps moving vehicles out of the background feature set,
        # including when the reference frame is first initialized. Detection
        # and tracking remain entirely in original source-frame coordinates.
        if self.camera_motion_enabled:
        	self.camera_transform = self.camera_motion.update(
        		frame.image,
        		moving_boxes=[t.bbox for t in tracks],
        		frame_id=getattr(frame, "idx", None),
        	)
        else:
        	self.camera_transform = np.array(
        		[[1., 0., 0.], [0., 1., 0.]], dtype=np.float32
    	)

        self.slots.set_camera_transform(self.camera_transform)

        # Release removed tracker IDs from parking state BEFORE processing
        # newly created tracks. This lets recover_lost_track() rebind a
        # replacement tracker to the original parked Vehicle identity.
        for t in removed:
            self.tripwire.forget(t.id)
            self.slots.on_track_removed(t)

            v = self.registry.by_track.get(t.id)
            if v is not None:
                v.pending.clear()
            self.registry.drop_track(t.id)

        self.last_tracks = tracks
        cc = self.cfg["color"]

        for t in tracks:
            # IMPORTANT:
            # Try to restore the original parked Vehicle identity BEFORE
            # get_or_create() can create a fresh Vehicle for this new tracker ID.
            recovered = self.slots.recover_lost_track(
                t,
                frame.ts,
                self.registry,
            )

            if recovered:
                log.info(
                    "Recovered parked vehicle identity for track=%s",
                    t.id,
                )

            # If this is a fresh tracker ID but we already have a recently
            # identified vehicle, preserve that Vehicle object so its finalized
            # plate/type/color can be used when parking is confirmed.
            v = self.registry.get_or_create(t, frame.ts)
            v.last_seen_ts = frame.ts
            v.cls_votes[t.cls] += 1

            x1, y1, x2, y2 = t.bbox

            if (
                frame.ts - v.last_color_ts >= cc["interval_s"]
                and (y2 - y1) >= cc["min_box_h"]
                and sum(v.color_votes.values()) < cc["max_samples"]
            ):
                crop = frame.image[
                    int(y1):int(y2),
                    int(x1):int(x2),
                ]

                col = dominant_color(crop) if crop.size else None

                if col:
                    v.color_votes[col] += 1

                v.last_color_ts = frame.ts

            # Use the existing tripwire's confirmed direction and original
            # crossing timestamp to generate gate ENTRY/EXIT events.
            crossing = self.tripwire.update(
                t.id,
                (x1 + x2) / 2,
                y2,
                frame.ts,
            )

            # Tripwire is used only as positional/directional evidence.
            # Parking state is authoritative for ENTRY/EXIT lifecycle events.
            # Tripwire confirms gate ENTRY/EXIT. SlotManager independently
            # owns PARK_START/PARK_END. Lifecycle events pass through
            # the existing plate-validation hold in _emit().
            if crossing:
                direction, crossing_ts = crossing

                if direction == "entry":
                    self._emit(
                        events,
                        "ENTRY",
                        v,
                        crossing_ts,
                        direction="entry",
                        sid=v.slot_id,
                        inferred=False,
                    )
                    v.pending_entry_received_at = None

                elif direction == "exit":
                    if v.has_parked:
                        v.pending_exit_crossing_ts = crossing_ts
                        log.info(
                            "EXIT_DEFERRED vehicle=%s crossing_ts=%.3f reason=parked",
                            v.uid, crossing_ts,
                        )
                    else:
                        self._emit(
                            events,
                            "EXIT",
                            v,
                            crossing_ts,
                            direction="exit",
                            sid=v.slot_id,
                            inferred=False,
                        )
                        v.pending_exit_crossing_ts = None

                log.debug(
                    "Tripwire crossing track=%s direction=%s ts=%.3f",
                    t.id,
                    direction,
                    crossing_ts,
                )
            self._maybe_submit_plate(t, v, frame)


        # Retry recovered crops before polling so their results can be applied
        # by the normal result handler below.
        self._maybe_submit_recovered_crops(frame.ts)
        self._apply_plate_results()

        # SlotManager owns parking-slot transitions. Preserve their event types
        # rather than incorrectly translating them into gate ENTRY/EXIT.
        for slot_event, v, et, sid, inferred in self.slots.update(
            tracks,
            frame.ts,
            self.registry,
        ):
            if slot_event not in ("PARK_START", "PARK_END"):
                log.warning(
                    "Ignoring unknown SlotManager event type: %s",
                    slot_event,
                )
                continue

            event_type = slot_event

            self._emit(
                events,
                event_type,
                v,
                et,
                sid=sid,
                inferred=inferred,
            )

            if event_type == "PARK_START":
                v.has_parked = True
            elif event_type == "PARK_END":
                v.has_parked = False
                exit_ts = v.pending_exit_crossing_ts
                if exit_ts is not None:
                    self._emit(
                        events,
                        "EXIT",
                        v,
                        exit_ts,
                        direction="exit",
                        sid=sid,
                        inferred=False,
                    )
                    v.pending_exit_crossing_ts = None

            if inferred:
                self.registry.release(v)

        self._release_ready_parking_events(events)

        # Preserve the existing late plate-update behavior.
        for v in list(self.registry.by_uid.values()):
            c = v.voter.consensus()

            if (
                v.any_emitted
                and v.plate_final
                and c
                and c.text != v.emitted_plate
            ):
                self._emit(
                    events,
                    "VEHICLE_UPDATE",
                    v,
                    frame.ts,
                )

        
        for ev in events:
            self._publish_event(ev)

        # Keep the latest vehicle votes and OCR state durable for unresolved
        # lifecycle events, including after their vehicle track has disappeared.
        self._persist_all_pending_events()

        done = time.perf_counter()

        self.stats.add("detect", (t1 - t0) * 1000)
        self.stats.add("track", (t2 - t1) * 1000)
        self.stats.add("total", (done - t0) * 1000)
        self.stats.add("e2e", (self.now() - frame.ts) * 1000)
        self.stats.frames += 1

        return events

    def flush(self, ts):
        events = []

        # Collect completed OCR jobs; the worker may already be stopped.
        self._apply_plate_results()
        self._release_ready_parking_events(events, final=True)

        for v in list(self.registry.by_uid.values()):
            v.pending.clear()

        for ev in events:
            self._publish_event(ev)

        # Do not clear unresolved records. They must remain recoverable.
        self._persist_all_pending_events()

        return events

    def draw(self, img):
        g = self.geo

        cv2.line(
            img,
            tuple(map(int, g.tripwire[0])),
            tuple(map(int, g.tripwire[1])),
            (0, 255, 255),
            2,
        )

        inverse_transform = cv2.invertAffineTransform(self.camera_transform)
        for s in g.slots:
            occ = self.slots.occupants.get(s["slot_id"])
            col = (0, 0, 255) if occ else (0, 200, 0)
            current_polygon = np.round(transform_points(s["polygon"], inverse_transform)).astype(np.int32)

            cv2.polylines(
                img,
                [current_polygon],
                True,
                col,
                2,
            )

            cv2.putText(
                img,
                s["slot_id"],
                tuple(map(int, current_polygon[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                1,
            )

        # Visible parking occupancy summary
        total_slots = len(g.slots)
        occupied_slots = [
            s["slot_id"]
            for s in g.slots
            if self.slots.occupants.get(s["slot_id"])
        ]
        occupied_count = len(occupied_slots)
        free_count = max(0, total_slots - occupied_count)

        cv2.rectangle(img, (10, 10), (560, 105), (20, 20, 20), -1)
        cv2.putText(
            img,
            f"SLOTS: {occupied_count}/{total_slots} occupied",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"AVAILABLE: {free_count}",
            (25, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if occupied_slots:
            print(
                "SLOT_SUMMARY occupied=%s/%s slots=%s"
                % (occupied_count, total_slots, ",".join(occupied_slots)),
                flush=True,
            )

        for t in self.last_tracks:
            v = self.registry.by_track.get(t.id)

            x1, y1, x2, y2 = map(int, t.bbox)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 120, 0),
                2,
            )

            c = v.voter.consensus() if v else None

            try:
                _n = __import__('time').time()
                _d = self.__dict__.setdefault('_lp', {})
                if _n - _d.get(t.id, 0) >= 1.0:
                    _d[t.id] = _n
                    try:
                        _k = __import__('sqlite3').connect('data/live_attrs.db', timeout=2)
                        _k.execute('create table if not exists live_attrs(ts real, track integer, plate text, colour text)')
                        _k.execute('insert into live_attrs values (?,?,?,?)', (_n, t.id, (c.text if c else None), (v.color() if v else None)))
                        _k.commit(); _k.close()
                    except Exception:
                        pass
                    print('LIVE #%s plate=%s colour=%s' % (t.id, (c.text if c else '?'), (v.color() if v else '')), flush=True)
            except Exception:
                pass
            label = (
                f"#{t.id} "
                f"{(c.text if c else '?')} "
                f"{v.color() if v else ''}"
            )

            cv2.putText(
                img,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return img
