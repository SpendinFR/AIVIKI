from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class JSONLLogger:
    """Simple thread-safe JSONL logger used by the architecture."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
