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
from typing import Any, Dict


class JSONLLogger:
    """
    Logger JSONL thread-safe des événements agent.
    Ecrit dans runtime/agent_events.jsonl + snapshots optionnels.
    """

    def __init__(self, path: str = "runtime/agent_events.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event_type: str, **fields: Any) -> None:
        rec = {
            "t": time.time(),
            "type": event_type,
            **fields,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def snapshot(self, name: str, payload: Dict[str, Any]) -> str:
        snap_dir = "runtime/snapshots"
        os.makedirs(snap_dir, exist_ok=True)
        ts = int(time.time())
        out = os.path.join(snap_dir, f"{ts}_{name}.json")
        with self._lock:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return out

    def rotate(self, keep_last: int = 5) -> None:
        # basique: ne supprime rien par défaut
        pass
