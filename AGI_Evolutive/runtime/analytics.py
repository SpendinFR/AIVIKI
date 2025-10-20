from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize


class EventPipeline:
    """Pipeline asynchrone pour traiter les événements loggés.

    Les gestionnaires enregistrés reçoivent une copie de l'événement brut sans
    modifier le flux d'écriture principal.
    """

    def __init__(
        self,
        *,
        handlers: Optional[Sequence[Callable[[Dict[str, Any]], None]]] = None,
        max_queue_size: int = 2048,
    ) -> None:
        self._handlers: List[Callable[[Dict[str, Any]], None]] = list(handlers or [])
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="event-pipeline", daemon=True)
        self._thread.start()

    def add_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._handlers.append(handler)

    def submit(self, event: Dict[str, Any]) -> None:
        if self._stop_event.is_set():
            return
        payload = dict(event)
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            # Politique simple: on évacue l'élément le plus ancien pour conserver la cadence.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                # si le queue est encore full, abandonner silencieusement
                return

    def close(self, *, wait: bool = True) -> None:
        self._stop_event.set()
        if wait and self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                for handler in self._handlers:
                    try:
                        handler(dict(event))
                    except Exception:
                        # un handler ne doit pas faire tomber toute la pipeline
                        continue
            finally:
                self._queue.task_done()


class RollingMetricAggregator:
    """Agrégateur simple pour calculer statistiques cumulées sur des champs numériques."""

    def __init__(self, fields: Iterable[str]) -> None:
        self._fields = list(fields)
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, float]] = {
            name: {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": float("-inf")}
            for name in self._fields
        }

    def __call__(self, event: Dict[str, Any]) -> None:
        with self._lock:
            for field in self._fields:
                value = event.get(field)
                if isinstance(value, (int, float)):
                    stats = self._stats[field]
                    stats["count"] += 1
                    stats["sum"] += float(value)
                    stats["min"] = min(stats["min"], float(value))
                    stats["max"] = max(stats["max"], float(value))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result: Dict[str, Dict[str, float]] = {}
            for name, stats in self._stats.items():
                count = stats["count"]
                avg = stats["sum"] / count if count else 0.0
                result[name] = {
                    "count": count,
                    "sum": stats["sum"],
                    "min": stats["min"] if count else 0.0,
                    "max": stats["max"] if count else 0.0,
                    "avg": avg,
                }
            return result


class SnapshotDriftTracker:
    """Compare les snapshots successifs et journalise les dérives."""

    def __init__(self, log_path: str = "runtime/snapshot_drifts.jsonl") -> None:
        self.log_path = log_path
        directory = os.path.dirname(self.log_path) or "."
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def record(self, snapshot_path: str, data: Dict[str, Any]) -> None:
        sanitized = json_sanitize(data or {})
        payload = json.dumps(sanitized, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        drift_event: Optional[Dict[str, Any]] = None
        if self._last_hash is None:
            drift_event = {
                "t": time.time(),
                "snapshot": snapshot_path,
                "initial": True,
                "changed_keys": [],
            }
        elif digest != self._last_hash:
            changed = sorted(self._diff_keys(self._last_snapshot or {}, sanitized))
            drift_event = {
                "t": time.time(),
                "snapshot": snapshot_path,
                "initial": False,
                "changed_keys": changed,
            }

        self._last_hash = digest
        self._last_snapshot = sanitized

        if drift_event is None:
            return

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(drift_event, ensure_ascii=False) + "\n")

    def _diff_keys(self, previous: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
        keys = set(previous.keys()) | set(current.keys())
        diffs: List[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                diffs.append(path)
                continue
            old_value = previous[key]
            new_value = current[key]
            if isinstance(old_value, dict) and isinstance(new_value, dict):
                diffs.extend(self._diff_keys(old_value, new_value, path))
            elif old_value != new_value:
                diffs.append(path)
        return diffs

