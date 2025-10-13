import time, json, os
from collections import deque


class Telemetry:
    def __init__(self, maxlen=2000):
        self.events = deque(maxlen=maxlen)
        self._jsonl_path = None
        self._console = False

    def enable_jsonl(self, path="logs/events.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._jsonl_path = path

    def enable_console(self, on=True):
        self._console = bool(on)

    def log(self, event_type, subsystem, data=None, level="info"):
        e = {
            "ts": time.time(),
            "type": event_type,
            "subsystem": subsystem,
            "level": level,
            "data": data or {}
        }
        self.events.append(e)
        # disque
        if self._jsonl_path:
            try:
                with open(self._jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # console légère
        if self._console and level in ("info", "warn", "error"):
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            print(f"[{ts}] {subsystem}/{level} {event_type} :: {e['data']}")

    def tail(self, n=50):
        return list(self.events)[-max(0, n):]

    def snapshot(self):
        by_sub = {}
        for e in self.events:
            by_sub[e["subsystem"]] = by_sub.get(e["subsystem"], 0) + 1
        return {"events_count": len(self.events), "events_by_subsystem": by_sub}
