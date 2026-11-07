import json
from collections import defaultdict
import random
from pathlib import Path

base = Path(__file__).resolve().parent / "audit_reference"
src = base / "dvg_test.jsonl"
out = base / "audit_60.jsonl"
want = {"traffic": 10, "productive": 10,
        "security": 10, "social": 10, "sport": 10, "disaster": 10}
groups = defaultdict(list)
with open(src, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        cls = row["class"].strip()
        if cls in want:
            groups[cls].append(row)
rows = []
rng = random.Random(42)
for cls in ["traffic", "productive", "security", "social", "sport", "disaster"]:
    rows.extend(rng.sample(groups[cls], want[cls]))
assert len(rows) == 60, {k: len(v) for k, v in groups.items()}
with open(out, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print({k: len(v) for k, v in groups.items()})
