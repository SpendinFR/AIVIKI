from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

ArchFactory = Callable[[Dict[str, Any]], Any]


class SandboxRunner:
    """Offline sandbox responsible for running evaluation suites."""

    def __init__(self, arch_factory: ArchFactory, eval_root: str = "data/eval") -> None:
        self.arch_factory = arch_factory
        self.eval_root = eval_root
        os.makedirs(self.eval_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation data loading helpers
    def _load_eval(self, name: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.eval_root, f"{name}.jsonl")
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    # ------------------------------------------------------------------
    # Individual evaluations
    def _eval_abduction(self, arch: Any) -> Tuple[List[Dict[str, float]], List[float]]:
        tasks = self._load_eval("abduction")
        if not tasks:
            tasks = [
                {"obs": "énigme simple: indice A & B", "gold": "hyp_a"},
                {"obs": "énigme simple: indice C", "gold": "hyp_c"},
            ]

        samples: List[Dict[str, float]] = []
        scores: List[float] = []
        for task in tasks:
            t0 = time.time()
            try:
                hyps = arch.abduction.generate(task.get("obs"))
            except Exception:
                hyps = []
            top = hyps[0].label if hyps else ""
            acc = 1.0 if top == task.get("gold") else 0.0
            dt = time.time() - t0
            samples.append({"acc": acc, "time": dt})
            scores.append(acc)
        return samples, scores

    def _eval_concepts(self, arch: Any) -> Tuple[List[Dict[str, float]], List[float]]:
        tasks = self._load_eval("concepts")
        if not tasks:
            tasks = [
                {
                    "concept": "principe_X",
                    "has_def": True,
                    "has_examples": True,
                    "has_counter": True,
                }
            ]

        samples: List[Dict[str, float]] = []
        scores: List[float] = []
        for task in tasks:
            concept = task.get("concept")
            learning = getattr(arch, "learning", None)
            if learning and hasattr(learning, "self_assess_concept"):
                try:
                    result = learning.self_assess_concept(concept)
                except Exception:
                    result = {"confidence": 0.0}
            else:
                result = {"confidence": 0.0}
            conf = float(result.get("confidence", 0.0))
            acc = 1.0 if conf >= 0.9 else 0.0
            samples.append({"acc": acc, "time": 0.0})
            scores.append(acc)
        return samples, scores

    # ------------------------------------------------------------------
    # Global run
    def run_all(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        arch = self.arch_factory(overrides)
        try:
            if hasattr(arch, "io") and hasattr(arch.io, "disable_network"):
                arch.io.disable_network()
        except Exception:
            pass

        acc_samples: List[Dict[str, float]] = []
        acc_scores: List[float] = []

        samples, scores = self._eval_abduction(arch)
        acc_samples.extend(samples)
        acc_scores.extend(scores)

        samples, scores = self._eval_concepts(arch)
        acc_samples.extend(samples)
        acc_scores.extend(scores)

        return {"samples": acc_samples, "scores": acc_scores}
