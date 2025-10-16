from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

from AGI_Evolutive.utils.jsonsafe import json_sanitize

if TYPE_CHECKING:
    from .quality import QualityGateRunner

class PromotionManager:
    """Manage candidate overrides and promotion history."""

    def __init__(self, root: str = "config") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "candidates"), exist_ok=True)
        self.active_path = os.path.join(self.root, "active_overrides.json")
        self.hist_path = os.path.join(self.root, "history.jsonl")

    # ------------------------------------------------------------------
    # Active overrides management
    def load_active(self) -> Dict[str, Any]:
        if os.path.exists(self.active_path):
            with open(self.active_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def save_active(self, overrides: Dict[str, Any]) -> None:
        with open(self.active_path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(overrides), handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Candidate lifecycle
    def stage_candidate(
        self,
        overrides: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        cid = str(uuid.uuid4())
        path = os.path.join(self.root, "candidates", f"{cid}.json")
        payload = {
            "overrides": overrides,
            "metrics": metrics,
            "metadata": metadata or {},
            "t": time.time(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(json_sanitize(payload), handle, ensure_ascii=False, indent=2)
        return cid

    def read_candidate(self, cid: str) -> Dict[str, Any]:
        path = os.path.join(self.root, "candidates", f"{cid}.json")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def promote(self, cid: str, quality_runner: Optional["QualityGateRunner"] = None) -> Dict[str, Any]:
        data = self.read_candidate(cid)
        quality_report: Optional[Dict[str, Any]] = None
        if quality_runner is not None:
            try:
                quality_report = quality_runner.run(data.get("overrides", {}))
            except Exception as exc:
                raise RuntimeError(f"quality_gate_error: {exc}") from exc
            if not quality_report.get("passed", False):
                raise RuntimeError("quality_gates_failed")
        self.save_active(data.get("overrides", {}))
        record = {
            "t": time.time(),
            "event": "promote",
            "cid": cid,
            "overrides": data.get("overrides"),
            "metrics": data.get("metrics"),
            "metadata": data.get("metadata"),
            "quality": quality_report,
        }
        with open(self.hist_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_sanitize(record)) + "\n")
        return record

    def rollback(self, steps: int = 1) -> None:
        if not os.path.exists(self.hist_path):
            return
        with open(self.hist_path, "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        prev: Dict[str, Any] | None = None
        for entry in reversed(lines):
            if entry.get("event") == "promote":
                if steps <= 0:
                    prev = entry
                    break
                steps -= 1
        if not prev:
            return
        self.save_active(prev.get("overrides", {}))
        with open(self.hist_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    json_sanitize({"t": time.time(), "event": "rollback", "to": prev})
                )
                + "\n"
            )
