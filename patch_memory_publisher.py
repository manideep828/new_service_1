from pathlib import Path
import re

p = Path("s1/main.py")
s = p.read_text(encoding="utf-8-sig")

pattern = r"(?m)^class MemoryPublisher:\s*$"
matches = list(re.finditer(pattern, s))

if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one MemoryPublisher class; found {len(matches)}. File unchanged."
    )

start = matches[0].end()
rest = s[start:]
if re.match(r"\s*\n[ \t]+local_preview\s*=", rest):
    print("local_preview already exists; no change needed.")
else:
    s = s[:start] + "\n    local_preview = True" + s[start:]
    p.write_text(s, encoding="utf-8")
    print("Added local_preview = True.")

print("\nVerification:")
for line in p.read_text(encoding="utf-8").splitlines():
    if line.strip() == "class MemoryPublisher:" or "local_preview" in line:
        print(line)
