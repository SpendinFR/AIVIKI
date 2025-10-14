import json
import os
import time
import uuid
from typing import Any, Dict, List

from core.config import cfg

_DIR = cfg()["MEM_DIR"]


class MemoryStore:
    """Simple JSON-file based episodic memory store."""

    def __init__(self) -> None:
        os.makedirs(_DIR, exist_ok=True)

    def add_memory(self, payload: Dict[str, Any]) -> str:
        memory_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
        path = os.path.join(_DIR, memory_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return memory_id

    def get_recent_memories(self, n: int = 20) -> List[Dict[str, Any]]:
        files = sorted(os.listdir(_DIR))[-n:]
        out: List[Dict[str, Any]] = []
        for filename in files:
            try:
                with open(os.path.join(_DIR, filename), "r", encoding="utf-8") as handle:
                    out.append(json.load(handle))
            except Exception:
                pass
        return out
