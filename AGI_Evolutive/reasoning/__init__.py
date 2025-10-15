"""Reasoning system responsible for producing structured hypothesis/test plans."""

import time
from collections import deque
from typing import Any, Dict, List, Optional

from .structures import Evidence, Hypothesis, Test, episode_record


class ReasoningSystem:
    """Generates structured reasoning episodes with traceable hypotheses and tests."""

    def __init__(self, architecture, memory_system=None, perception_system=None):
        self.arch = architecture
        self.memory = memory_system
        self.perception = perception_system

        self.reasoning_history: Dict[str, Any] = {
            "recent_inferences": deque(maxlen=200),
            "learning_trajectory": [],
            "errors": deque(maxlen=200),
            "stats": {
                "n_episodes": 0,
                "avg_confidence": 0.50,
                "strategy_preferences": {
                    "abduction": 0.33,
                    "deduction": 0.34,
                    "analogy": 0.33,
                },
            },
        }

    # ------------------- API publique -------------------
    def reason_about(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Réalise un raisonnement structuré sur *prompt* et retourne un épisode détaillé."""
        t0 = time.time()

        strategy = self._pick_strategy(prompt)
        hypos = self._make_hypotheses(prompt, strategy=strategy)
        scores = self._score_hypotheses(prompt, hypos, strategy)
        chosen_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0

        tests = self._propose_tests(prompt, strategy=strategy)
        note = self._simulate_micro_inference(prompt, strategy, hypos[chosen_idx])
        evidence = Evidence(notes=note, confidence=min(0.55 + 0.15 * scores[chosen_idx], 0.95))

        reasoning_time = max(0.05, time.time() - t0)
        complexity = self._estimate_complexity(prompt, strategy, hypos)
        final_conf = float(min(0.9, 0.45 + 0.4 * scores[chosen_idx] + 0.10 * (1.0 - complexity)))

        ep = episode_record(
            user_msg=prompt,
            hypotheses=hypos,
            chosen_index=chosen_idx,
            tests=tests,
            evidence=evidence,
            result_text=evidence.notes,
            final_confidence=final_conf,
        )
        ep["reasoning_time"] = reasoning_time
        ep["complexity"] = complexity
        ep["strategy"] = strategy

        self._push_episode(ep)

        summary = self._make_readable_summary(strategy, hypos, chosen_idx, tests, evidence, final_conf)
        self._learn(final_conf, strategy)

        try:
            if hasattr(self.arch, "logger") and self.arch.logger:
                self.arch.logger.write("reasoning.episode", episode=ep)
        except Exception:
            pass

        return {
            "summary": summary,
            "chosen_hypothesis": hypos[chosen_idx].content if hypos else "",
            "tests": [t.description for t in tests],
            "final_confidence": final_conf,
            "appris": [
                f"Stratégie={strategy}, complexité≈{complexity:.2f}",
                "Toujours relier hypothèse→test→évidence (traçabilité).",
            ],
            "prochain_test": tests[0].description if tests else "-",
            "episode": ep,
        }

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Résumé statistique utilisé par la métacognition."""
        stats = self.reasoning_history["stats"]
        recents = list(self.reasoning_history["recent_inferences"])
        if recents:
            stats["avg_confidence"] = sum(x.get("final_confidence", 0.5) for x in recents) / len(recents)
        return {
            "episodes": stats["n_episodes"],
            "average_confidence": float(stats.get("avg_confidence", 0.5)),
            "strategy_preferences": dict(stats.get("strategy_preferences", {})),
        }

    # ------------------- internes -------------------
    def _pick_strategy(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(keyword in lowered for keyword in ["pourquoi", "why", "cause"]):
            return "abduction"
        if any(keyword in lowered for keyword in ["comment", "plan", "steps", "étapes"]):
            return "deduction"
        return "analogy"

    def _make_hypotheses(self, prompt: str, strategy: str) -> List[Hypothesis]:
        base = [
            Hypothesis(content="tu veux une explication avec étapes et tests", prior=0.55),
            Hypothesis(content="tu veux que j'auto-documente ce que j'apprends", prior=0.50),
            Hypothesis(content="tu veux un patch/exécution immédiate", prior=0.45),
        ]
        if strategy == "analogy":
            base.append(Hypothesis(content="tu veux comparer avec un cas passé", prior=0.48))
        return base

    def _score_hypotheses(self, prompt: str, hypos: List[Hypothesis], strategy: str) -> List[float]:
        lowered = prompt.lower()
        scores: List[float] = []
        for hypo in hypos:
            score = hypo.prior
            if "test" in lowered and "test" in hypo.content:
                score += 0.1
            if strategy == "abduction" and "pourquoi" in lowered:
                score += 0.1
            scores.append(min(1.0, score))
        return scores

    def _propose_tests(self, prompt: str, strategy: str) -> List[Test]:
        base_tests = [
            Test(description="Formuler 2 hypothèses alternatives et demander validation", cost_est=0.3, expected_information_gain=0.55),
            Test(description="Rechercher 2 exemples récents similaires", cost_est=0.2, expected_information_gain=0.45),
        ]
        if strategy == "deduction":
            base_tests.append(
                Test(
                    description="Détailler un plan en 3 étapes avec vérification de chaque sous-résultat",
                    cost_est=0.35,
                    expected_information_gain=0.6,
                )
            )
        return base_tests

    def _simulate_micro_inference(self, prompt: str, strategy: str, hypothesis: Hypothesis) -> str:
        if strategy == "abduction":
            return f"Hypothèse choisie: {hypothesis.content}. Cause probable: manque de contexte explicite."
        if strategy == "deduction":
            return f"Plan issu de l'hypothèse '{hypothesis.content}': découper la tâche, tester chaque étape et consigner."
        return f"En comparant avec des cas passés, '{hypothesis.content}' semble le plus prometteur."

    def _estimate_complexity(self, prompt: str, strategy: str, hypos: List[Hypothesis]) -> float:
        length_factor = min(1.0, len(prompt) / 500.0)
        strategy_factor = 0.6 if strategy == "deduction" else 0.5
        hypo_factor = min(1.0, 0.2 * len(hypos))
        return float(min(1.0, 0.3 + length_factor * 0.4 + strategy_factor * 0.2 + hypo_factor * 0.2))

    def _make_readable_summary(
        self,
        strategy: str,
        hypos: List[Hypothesis],
        chosen_idx: int,
        tests: List[Test],
        evidence: Evidence,
        final_conf: float,
    ) -> str:
        hypo_text = hypos[chosen_idx].content if hypos else "hypothèse principale"
        test_text = tests[0].description if tests else "observer le prochain signal"
        return (
            f"Stratégie {strategy}: retenir '{hypo_text}'. "
            f"Test prioritaire: {test_text}. Évidence: {evidence.notes}. "
            f"Confiance finale≈{final_conf:.2f}."
        )

    def _push_episode(self, episode: Dict[str, Any]) -> None:
        history = self.reasoning_history
        history["recent_inferences"].append(
            {
                "final_confidence": episode.get("final_confidence", 0.5),
                "strategy": episode.get("strategy"),
                "ts": time.time(),
            }
        )
        history["stats"]["n_episodes"] += 1

    def _learn(self, confidence: float, strategy: str) -> None:
        self.reasoning_history["learning_trajectory"].append({"ts": time.time(), "confidence": confidence})
        prefs = self.reasoning_history["stats"].setdefault("strategy_preferences", {})
        prefs[strategy] = prefs.get(strategy, 0.0) * 0.8 + 0.2
