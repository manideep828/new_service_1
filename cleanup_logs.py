from pathlib import Path

p = Path("s1/logic.py")
s = p.read_text(encoding="utf-8")
old = s
s = s.replace('log.info(\n                            "PARK_START_WAIT', 'log.debug(\n                            "PARK_START_WAIT')
s = s.replace('log.info(\n                            "PARK_START_WAIT track=', 'log.debug(\n                            "PARK_START_WAIT track=')
if s == old:
    print("No PARK_START_WAIT logging lines matched; logic.py unchanged.")
else:
    p.write_text(s, encoding="utf-8")
    print("Changed matching PARK_START_WAIT messages to DEBUG.")
