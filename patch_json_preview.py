from pathlib import Path
import re

p = Path(r"s1/main.py")
s = p.read_text(encoding="utf-8")

if "import json" not in s:
    s = s.replace("import logging", "import json\nimport logging", 1)

pattern = r'    def publish\(self, event\):\r?\n.*?(?=    def save_pending_event\(self, event_key, payload\):)'
replacement = '''    def publish(self, event):
        """Print the outgoing event payload locally; never contact Service 2."""
        print(
            "\\\\n========== SERVICE 2 JSON PREVIEW (NOT SENT) ==========",
            flush=True,
        )
        print(
            json.dumps(event, indent=2, ensure_ascii=False, default=str),
            flush=True,
        )
        print("========== END JSON PREVIEW ===========================\\\\n", flush=True)

'''
updated, count = re.subn(pattern, replacement, s, count=1, flags=re.S)
if count != 1:
    raise SystemExit(
        "PATCH ABORTED: expected MemoryPublisher.publish was not found. "
        "Original file is unchanged."
    )

p.write_text(updated, encoding="utf-8")
print("Updated MemoryPublisher.publish")
