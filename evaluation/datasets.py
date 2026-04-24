"""Common evaluation format. Write one converter per benchmark into this shape.

One JSON object per line:
    {"id": "...",
     "text": "free-text patient description",
     "symptoms": [{"cui": "C0015967", "negated": false}, ...],   # optional
     "diagnosis": "C0021400"}                                    # optional
"""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
