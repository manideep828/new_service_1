from pathlib import Path
import re

p = Path("s1/pipeline.py")
s = p.read_text(encoding="utf-8")

if "local_preview = getattr(self.publisher, " not in s:
    marker = "        now = self.now()\n        remaining = []\n"
    if marker not in s:
        raise SystemExit("PATCH ABORTED: release method marker not found.")
    s = s.replace(
        marker,
        marker + "        local_preview = getattr(self.publisher, 'local_preview', False)\n",
        1,
    )

old = '''            if now >= pending["deadline"] and not pending["timed_out"]:
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
'''

new = '''            if local_preview and (
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
'''

if old not in s:
    raise SystemExit(
        "PATCH ABORTED: expected timeout block not found. No patched file written."
    )

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("Patched local-only lifecycle event release.")
