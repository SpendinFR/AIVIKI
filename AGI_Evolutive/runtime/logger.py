from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

from AGI_Evolutive.utils.jsonsafe import json_sanitize


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
        line = json.dumps(json_sanitize(rec), ensure_ascii=False)
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
                json.dump(json_sanitize(payload), f, ensure_ascii=False, indent=2)
        return out

    def rotate(self, keep_last: int = 5) -> None:
        if keep_last is not None and keep_last < 0:
            raise ValueError("keep_last doit être >= 0")

        # basique: ne supprime rien par défaut
        if not os.path.exists(self.path):
            return

        directory = os.path.dirname(self.path) or "."
        basename = os.path.basename(self.path)
        timestamp = int(time.time())
        rotated_path = os.path.join(directory, f"{basename}.{timestamp}")

        with self._lock:
            if not os.path.exists(self.path):
                return

            # éviter collisions si plusieurs rotations dans la même seconde
            suffix = 0
            candidate = rotated_path
            while os.path.exists(candidate):
                suffix += 1
                candidate = os.path.join(directory, f"{basename}.{timestamp}.{suffix}")
            os.replace(self.path, candidate)

            # créer un nouveau fichier vide pour le log courant
            open(self.path, "a", encoding="utf-8").close()

            if keep_last is None:
                return

            # supprimer les plus anciens fichiers archivés au-delà de la limite
            entries = []
            for name in os.listdir(directory):
                if name == basename:
                    continue
                if name.startswith(f"{basename}."):
                    full_path = os.path.join(directory, name)
                    entries.append((os.path.getmtime(full_path), full_path))

            entries.sort(reverse=True)
            for _, path in entries[keep_last:]:
                try:
                    os.remove(path)
                except OSError:
                    # ne pas interrompre la rotation si suppression impossible
                    pass
