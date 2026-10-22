import sys, json, traceback, logging
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(name)s: %(message)s")
from s1.config import load_config
from s1.main import run, MemoryPublisher

VIDEO = r"C:\Users\adity\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\FBD12DC6D50FEA5B35FAA87B4E26C7DBA804C3D7\transfers\2026-40\VN20261005_103538.mp4"

class Capture(MemoryPublisher):
    events = []
    def publish(self, e):
        Capture.events.append(e)
        v = e.get("vehicle") or {}
        import json
        print("\n" + "=" * 72)
        print("SERVICE 1 EVENT")
        print("=" * 72)
        print(json.dumps(e, indent=2, ensure_ascii=False, default=str))
        print("=" * 72)
        super().publish(e)

cfg = load_config("config/config.yaml", {"source": VIDEO, "replay": "sync", "geometry_file": "config/geometry.json",
                                         "parking": {"park_dwell_s": 4.0}})
print("EFFECTIVE_PARKING_CFG:", cfg["parking"])
try:
    run(cfg, publisher=Capture())
except Exception:
    traceback.print_exc(file=sys.stdout)
print("\n" + "=" * 72)
print("FINAL SERVICE 1 TEST SUMMARY")
print("=" * 72)

for i, e in enumerate(Capture.events, 1):
    v = e.get("vehicle") or {}
    parking = e.get("parking") or {}
    print(f"""
EVENT #{i}
  event_type   : {e.get("event_type")}
  event_id     : {e.get("event_id")}
  plate        : {v.get("plate")}
  vehicle_type : {v.get("type")}
  color        : {v.get("color")}
  slot_id      : {parking.get("slot_id")}
  event_time   : {e.get("event_time") or e.get("timestamp")}
""".rstrip())

print("-" * 72)
print("TOTAL EVENTS :", len(Capture.events))
print("=" * 72)

