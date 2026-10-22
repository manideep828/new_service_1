import argparse
import json
import logging
import time
from collections import Counter

import cv2

from .config import load_config
from .yolo_detector import YoloDetector
from .geometry import Geometry
from .grabber import open_source
from .outbox import Outbox
from .pipeline import Pipeline
from .s2_mapper import S2Mapper
from .uploader import UploadingOutbox
from .plates import PlateReader, PlateWorker

log = logging.getLogger("main")


class MemoryPublisher:
    local_preview = True
    """In-memory publisher for local video testing.

    This deliberately does not contact Service 2 and does not write to
    data/outbox.db. It provides the small publisher interface Pipeline needs.
    """

    def __init__(self):
        self.pending_events = {}

    def publish(self, event):
        """Print the outgoing event payload locally; never contact Service 2."""
        print(
            "\n========== SERVICE 2 JSON PREVIEW (NOT SENT) ==========",
            flush=True,
        )
        print(
            json.dumps(event, indent=2, ensure_ascii=False, default=str),
            flush=True,
        )
        print("========== END JSON PREVIEW ===========================\n", flush=True)

    def save_pending_event(self, event_key, payload):
        self.pending_events[event_key] = payload

    def load_pending_events(self):
        return list(self.pending_events.items())

    def delete_pending_event(self, event_key):
        self.pending_events.pop(event_key, None)

    def stop(self, wait=True, timeout=5.0):
        return True


def run(
    cfg,
    show=False,
    detector=None,
    publisher=None,
    plate_worker=None,
    source=None,
    no_publisher=False,
):
    # Initialize models before starting video capture.
    d = cfg["detector"]
    detector = detector or YoloDetector(
        d["model_path"],
        d["input_size"],
        d["score_thr"],
        d["nms_thr"],
        d["classes"],
        d["threads"],
    )


    source = source or open_source(cfg)

    first = None
    while first is None:
        first = source.read(-1, timeout=2.0)
        if first is None and getattr(source, "eof", False):
            raise SystemExit("could not read any frame from the source")
        if first is None:
            log.info("waiting for first frame...")

    H, W = first.image.shape[:2]
    log.info("source ready: %dx%d", W, H)

    publisher_was_created = publisher is None

    if publisher_was_created:
        if no_publisher:
            publisher = MemoryPublisher()
            log.info(
                "LOCAL TEST MODE: publisher disabled; "
                "events will not be sent to Service 2"
            )
        else:
            publisher = S2Mapper(
                UploadingOutbox(
                    cfg["publisher"]["outbox_db"],
                    cfg["publisher"]["s2_url"],
                    timeout=30.0,
                )
            )


    if plate_worker is None and cfg["plate"]["enabled"]:
        p = cfg["plate"]
        plate_worker = PlateWorker(
            PlateReader(p["detector_model"], p["ocr_model"])
        )

    geometry = Geometry.load(cfg["geometry_file"])
    pipe = Pipeline(
        cfg,
        geometry,
        (W, H),
        detector,
        publisher,
        plate_worker,
    )

    min_dt = 1.0 / d["max_fps"]
    last_proc_ts, last_idx, last_report = (
        -1e18,
        first.idx - 1,
        time.time(),
    )
    frame = first
    event_counts = Counter()

    if show:
        cv2.namedWindow("garage-s1", cv2.WINDOW_NORMAL)
        cv2.imshow("garage-s1", frame.image)
        log.info("Preview ready. Press SPACE to start, or Q to quit.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                source.stop()
                if plate_worker is not None:
                    plate_worker.stop(wait=True)
                cv2.destroyAllWindows()
                return

    try:
        while frame is not None:
            last_idx = frame.idx

            if frame.ts - last_proc_ts >= min_dt * 0.9:
                last_proc_ts = frame.ts

                for ev in pipe.process(frame):
                    event_type = ev["event_type"]
                    event_counts[event_type] += 1

                    log.info(
                        "%s %s plate=%s bay=%s t=%s",
                        event_type,
                        ev.get("direction") or "",
                        (ev.get("plate") or {}).get("text"),
                        (ev.get("bay") or {}).get("id"),
                        ev.get("event_time"),
                    )

                if show:
                    display_frame = pipe.draw(frame.image.copy())

                    # Fit preview to the screen without changing detection coordinates.
                    screen_w, screen_h = 1100, 800
                    frame_h, frame_w = display_frame.shape[:2]
                    scale = min(
                        screen_w / frame_w,
                        screen_h / frame_h,
                        1.0,
                    )

                    if scale < 1.0:
                        display_frame = cv2.resize(
                            display_frame,
                            (int(frame_w * scale), int(frame_h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    cv2.imshow("garage-s1", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if time.time() - last_report > 5:
                log.info(pipe.stats.summary())
                last_report = time.time()

            frame = source.read(last_idx, timeout=1.0)

            if frame is None and not getattr(source, "eof", False):
                log.warning("no new frame for 1s (camera stalled?)")
                frame = source.read(last_idx, timeout=5.0)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            source.stop()
        except Exception:
            log.exception("failed to stop source")

        try:
            if plate_worker is not None:
                stopped = plate_worker.stop(wait=True)

                if not stopped:
                    log.error(
                        "plate worker did not stop; "
                        "OCR results may be incomplete"
                    )

        except Exception:
            log.exception("failed to stop plate worker")

        try:
            flushed_events = pipe.flush(time.time())
            for ev in flushed_events:
                event_type = ev["event_type"]
                event_counts[event_type] += 1
                log.info(
                    "%s %s plate=%s bay=%s t=%s",
                    event_type,
                    ev.get("direction") or "",
                    (ev.get("plate") or {}).get("text"),
                    (ev.get("bay") or {}).get("id"),
                    ev.get("event_time"),
                )

            log.info(
                "EVENT SUMMARY: ENTRY=%d PARK_START=%d PARK_END=%d EXIT=%d TOTAL=%d",
                event_counts["ENTRY"],
                event_counts["PARK_START"],
                event_counts["PARK_END"],
                event_counts["EXIT"],
                sum(event_counts.values()),
            )

        except Exception:
            log.exception("pipeline flush failed")

        if publisher_was_created:
            try:
                stopped = publisher.stop(wait=True)

                if not stopped:
                    log.error(
                        "publisher did not stop; "
                        "delivery may continue in the background"
                    )

            except Exception:
                log.exception("failed to stop publisher")

        log.info(pipe.stats.summary())


def cli():
    ap = argparse.ArgumentParser(
        description="Garage vision service (S1)"
    )

    ap.add_argument(
        "--config",
        default="config/config.yaml",
    )

    ap.add_argument(
        "--source",
        help="video file or rtsp:// URL (overrides config)",
    )

    ap.add_argument(
        "--replay",
        choices=["realtime", "sync"],
        help="file replay mode",
    )

    ap.add_argument(
        "--show",
        action="store_true",
        help="show debug window (press q to quit)",
    )

    ap.add_argument(
        "--no-plates",
        action="store_true",
        help="disable plate reading (speed test)",
    )

    ap.add_argument(
        "--no-publisher",
        action="store_true",
        help="disable Service 2 publishing for local video testing",
    )

    a = ap.parse_args()

    over = {}

    if a.source:
        over["source"] = a.source

    if a.replay:
        over["replay"] = a.replay

    if a.no_plates:
        over["plate"] = {"enabled": False}

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(
        load_config(a.config, over),
        show=a.show,
        no_publisher=a.no_publisher,
    )


if __name__ == "__main__":
    cli()

