from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .metrics import aggregate_metrics, bootstrap_superiority, dominates
from .mutations import generate_overrides
from .promote import PromotionManager
from .sandbox import SandboxRunner, ArchFactory


class SelfImprover:
    """Coordinate champion/challenger evaluation and promotion workflow."""

    def __init__(
        self,
        arch_factory: ArchFactory,
        memory: Any,
        question_manager: Optional[Any] = None,
        config_root: str = "config",
        eval_root: str = "data/eval",
    ) -> None:
        self.arch_factory = arch_factory
        self.memory = memory
        self.questions = question_manager
        self.prom = PromotionManager(config_root)
        self.sandbox = SandboxRunner(arch_factory, eval_root=eval_root)

    # ------------------------------------------------------------------
    # Internal helpers
    def _active_overrides(self) -> Dict[str, Any]:
        return self.prom.load_active()

    def _log_experiment(self, payload: Dict[str, Any]) -> None:
        os.makedirs("data/self_improve", exist_ok=True)
        record = {"t": time.time(), **payload}
        with open("data/self_improve/experiments.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Public API
    def run_cycle(self, n_candidates: int = 4) -> Optional[str]:
        base = self._active_overrides()
        champion_eval = self.sandbox.run_all(base)
        champ_aggr = aggregate_metrics(champion_eval.get("samples", []))
        champ_scores = champion_eval.get("scores", [])
        self._log_experiment({"kind": "champion_eval", "metrics": champ_aggr})

        candidates = generate_overrides(base, n=n_candidates)
        best_cid: Optional[str] = None
        best_metrics: Optional[Dict[str, float]] = None

        for cand in candidates:
            evaluation = self.sandbox.run_all(cand)
            aggregated = aggregate_metrics(evaluation.get("samples", []))
            scores = evaluation.get("scores", [])
            p_value = bootstrap_superiority(champ_scores, scores)
            self._log_experiment(
                {
                    "kind": "challenger_eval",
                    "overrides": cand,
                    "metrics": aggregated,
                    "p": p_value,
                }
            )
            if dominates(champ_aggr, aggregated) and p_value < 0.2:
                cid = self.prom.stage_candidate(cand, aggregated)
                best_cid = cid
                best_metrics = aggregated

        if not best_cid or not best_metrics:
            try:
                if hasattr(self.memory, "add_memory"):
                    self.memory.add_memory(
                        kind="self_improve",
                        content="Aucun challenger ne surpasse le champion",
                        metadata={"champion": champ_aggr},
                    )
            except Exception:
                pass
            return None

        question = (
            f"Promotion du challenger {best_cid} ? "
            f"(acc={best_metrics.get('acc', 0.0):.3f}, "
            f"ece={best_metrics.get('cal_ece', 0.0):.3f}) "
            "Réponds 'oui' pour promouvoir."
        )
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    kind="promotion_request",
                    content=question,
                    metadata={"cid": best_cid, "metrics": best_metrics},
                )
            if self.questions and hasattr(self.questions, "add_question"):
                self.questions.add_question(question)
        except Exception:
            pass
        return best_cid

    def try_promote_from_reply(self, user_text: str) -> bool:
        normalized = (user_text or "").strip().lower()
        if normalized not in {"oui", "yes", "go", "ok"}:
            return False
        cid: Optional[str] = None
        if hasattr(self.memory, "get_recent_memories"):
            try:
                recents = self.memory.get_recent_memories(60)
            except Exception:
                recents = []
        else:
            recents = []
        for item in reversed(recents):
            if item.get("kind") == "promotion_request":
                meta = item.get("metadata") or {}
                cid = meta.get("cid")
                if cid:
                    break
        if not cid:
            return False
        self.promote(cid)
        return True

    def promote(self, cid: str) -> None:
        self.prom.promote(cid)
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(kind="promotion", content=f"Promu {cid}")
        except Exception:
            pass

    def rollback(self, steps: int = 1) -> None:
        self.prom.rollback(steps)
        try:
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(kind="rollback", content=f"Rollback {steps}")
        except Exception:
            pass
