"""Reasoning system responsible for producing structured hypothesis/test plans."""

import math
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .structures import Evidence, Hypothesis, Test, episode_record


class OnlineLinear:
    """Simple online generalized linear model with bounded weights."""

    def __init__(
        self,
        feature_names: Sequence[str],
        learning_rate: float = 0.15,
        l2: float = 0.01,
        weight_bounds: Sequence[float] = (0.0, 1.5),
    ) -> None:
        self.feature_names = list(feature_names)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.low, self.high = weight_bounds
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}

    def predict(self, features: Dict[str, float]) -> float:
        z = sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in self.feature_names)
        # Logistic squashing to keep score in [0, 1]
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, features: Dict[str, float], target: float) -> float:
        prediction = self.predict(features)
        error = prediction - target
        for name in self.feature_names:
            value = features.get(name, 0.0)
            grad = error * value + self.l2 * self.weights.get(name, 0.0)
            self.weights[name] = float(min(self.high, max(self.low, self.weights.get(name, 0.0) - self.learning_rate * grad)))
        return prediction


class ThompsonSelector:
    """Discrete Thompson Sampling helper for adaptively choosing options."""

    def __init__(self, option_keys: Sequence[str]) -> None:
        self.params: Dict[str, Dict[str, float]] = {
            key: {"alpha": 1.0, "beta": 1.0} for key in option_keys
        }

    def draw(self, option: str) -> float:
        params = self.params.setdefault(option, {"alpha": 1.0, "beta": 1.0})
        return random.betavariate(params["alpha"], params["beta"])

    def sample(self) -> str:
        return max(
            self.params,
            key=lambda key: self.draw(key),
        )

    def update(self, option: str, reward: float) -> None:
        if option not in self.params:
            self.params[option] = {"alpha": 1.0, "beta": 1.0}
        # Reward expected in [0, 1]
        self.params[option]["alpha"] += max(0.0, min(1.0, reward))
        self.params[option]["beta"] += max(0.0, min(1.0, 1.0 - reward))


class ReasoningSystem:
    """Generates structured reasoning episodes with traceable hypotheses and tests."""

    def __init__(self, architecture, memory_system=None, perception_system=None):
        self.arch = architecture
        self.memory = memory_system
        self.perception = perception_system

        self._feature_names = [
            "bias",
            "prompt_len",
            "contains_question",
            "contains_test",
            "strategy_match",
            "hypothesis_prior",
        ]
        self.strategy_models: Dict[str, OnlineLinear] = {
            strategy: OnlineLinear(self._feature_names)
            for strategy in ("abduction", "deduction", "analogy")
        }

        self._test_templates = self._build_test_templates()
        self.test_bandits: Dict[str, ThompsonSelector] = {}
        for strategy in ("abduction", "deduction", "analogy"):
            templates = list(self._test_templates.get("common", [])) + list(
                self._test_templates.get(strategy, [])
            )
            option_keys = [self._test_option_key(strategy, t["key"]) for t in templates]
            self.test_bandits[strategy] = ThompsonSelector(option_keys)

        smoothing_keys = ["beta_0.2", "beta_0.4", "beta_0.6", "beta_0.8"]
        self.preference_smoother = ThompsonSelector(smoothing_keys)
        self._current_smoothing_key: Optional[str] = None

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

        smoothing_value = self._select_smoothing_factor()
        strategy = self._pick_strategy(prompt)
        hypos = self._make_hypotheses(prompt, strategy=strategy)
        scores, features_by_idx = self._score_hypotheses(prompt, hypos, strategy)
        chosen_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0

        tests, chosen_test_keys = self._propose_tests(prompt, strategy=strategy)
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
        ep["hypothesis_features"] = features_by_idx
        ep["smoothing_value"] = smoothing_value
        ep["test_keys"] = chosen_test_keys

        self._push_episode(ep)

        summary = self._make_readable_summary(strategy, hypos, chosen_idx, tests, evidence, final_conf)
        self._learn(final_conf, strategy, features_by_idx.get(chosen_idx, {}), smoothing_value, chosen_test_keys)

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

    def _score_hypotheses(
        self, prompt: str, hypos: List[Hypothesis], strategy: str
    ) -> Tuple[List[float], Dict[int, Dict[str, float]]]:
        lowered = prompt.lower()
        scores: List[float] = []
        features_by_idx: Dict[int, Dict[str, float]] = {}
        for idx, hypo in enumerate(hypos):
            features = self._extract_features(lowered, hypo, strategy)
            model = self.strategy_models[strategy]
            learned_score = model.predict(features)
            # Blend learned score with prior for stability
            score = 0.55 * learned_score + 0.45 * min(1.0, max(0.0, hypo.prior))
            # Encourage hypotheses aligned with explicit cues
            if "test" in lowered and "test" in hypo.content:
                score += 0.05
            features_by_idx[idx] = features
            scores.append(min(1.0, score))
        return scores, features_by_idx

    def _propose_tests(self, prompt: str, strategy: str) -> Tuple[List[Test], List[str]]:
        lowered = prompt.lower()
        templates = list(self._test_templates.get("common", [])) + list(self._test_templates.get(strategy, []))
        bandit_key = strategy
        if bandit_key not in self.test_bandits:
            option_keys = [self._test_option_key(strategy, t["key"]) for t in templates]
            self.test_bandits[bandit_key] = ThompsonSelector(option_keys)
        bandit = self.test_bandits[bandit_key]

        scored_templates = []
        for template in templates:
            option_key = self._test_option_key(strategy, template["key"])
            posterior_sample = bandit.draw(option_key)
            keyword_bonus = 0.05 if any(word in lowered for word in template.get("keywords", [])) else 0.0
            score = posterior_sample + keyword_bonus + template.get("base_gain", 0.0)
            scored_templates.append((score, template, option_key))

        scored_templates.sort(key=lambda x: x[0], reverse=True)
        selected = scored_templates[:3]
        tests: List[Test] = []
        chosen_keys: List[str] = []
        for _, template, option_key in selected:
            tests.append(
                Test(
                    description=template["description"],
                    cost_est=template["cost_est"],
                    expected_information_gain=min(1.0, max(0.0, template.get("expected_information_gain", 0.5))),
                )
            )
            chosen_keys.append(option_key)
        return tests, chosen_keys

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

    def _learn(
        self,
        confidence: float,
        strategy: str,
        features: Dict[str, float],
        smoothing_value: float,
        test_keys: List[str],
    ) -> None:
        self.reasoning_history["learning_trajectory"].append({"ts": time.time(), "confidence": confidence})
        stats = self.reasoning_history["stats"]
        prefs = stats.setdefault("strategy_preferences", {})
        for strat in self.strategy_models.keys():
            prefs.setdefault(strat, 1.0 / len(self.strategy_models))

        baseline = stats.get("avg_confidence", 0.5)

        # Update strategy preference with adaptive smoothing
        old_pref = prefs.get(strategy, baseline)
        prefs[strategy] = (1.0 - smoothing_value) * old_pref + smoothing_value * confidence
        for strat in prefs:
            if strat != strategy:
                prefs[strat] *= 1.0 - 0.05 * smoothing_value
        self._normalize_preferences(prefs)

        stats["avg_confidence"] = (1.0 - smoothing_value) * baseline + smoothing_value * confidence

        if features:
            model = self.strategy_models[strategy]
            model.update(features, confidence)

        smoothing_key = self._current_smoothing_key
        if smoothing_key:
            reward = 1.0 if confidence >= baseline else 0.0
            self.preference_smoother.update(smoothing_key, reward)
        self._current_smoothing_key = None

        reward_conf = max(0.0, min(1.0, confidence))
        bandit = self.test_bandits.get(strategy)
        if bandit:
            for key in test_keys:
                bandit.update(key, reward_conf)

    # ------------------- helpers adaptatifs -------------------
    def _select_smoothing_factor(self) -> float:
        key = self.preference_smoother.sample()
        self._current_smoothing_key = key
        try:
            return float(key.split("_")[1])
        except (IndexError, ValueError):
            return 0.4

    def _test_option_key(self, strategy: str, key: str) -> str:
        return f"{strategy}:{key}"

    def _build_test_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "common": [
                {
                    "key": "alt_hypotheses",
                    "description": "Formuler 2 hypothèses alternatives et demander validation",
                    "cost_est": 0.3,
                    "expected_information_gain": 0.55,
                    "base_gain": 0.25,
                    "keywords": ["hypothèse", "pourquoi"],
                },
                {
                    "key": "retrieve_examples",
                    "description": "Rechercher 2 exemples récents similaires",
                    "cost_est": 0.2,
                    "expected_information_gain": 0.45,
                    "base_gain": 0.2,
                    "keywords": ["exemple", "analogie"],
                },
            ],
            "abduction": [
                {
                    "key": "root_cause",
                    "description": "Identifier la cause racine supposée et vérifier un symptôme clé",
                    "cost_est": 0.28,
                    "expected_information_gain": 0.6,
                    "base_gain": 0.3,
                    "keywords": ["cause", "pourquoi"],
                }
            ],
            "deduction": [
                {
                    "key": "step_plan",
                    "description": "Détailler un plan en 3 étapes avec vérification de chaque sous-résultat",
                    "cost_est": 0.35,
                    "expected_information_gain": 0.62,
                    "base_gain": 0.35,
                    "keywords": ["plan", "étapes", "comment"],
                },
                {
                    "key": "assertion_check",
                    "description": "Lister les hypothèses critiques et prévoir un test de non-régression",
                    "cost_est": 0.32,
                    "expected_information_gain": 0.58,
                    "base_gain": 0.28,
                    "keywords": ["test", "vérifier"],
                },
            ],
            "analogy": [
                {
                    "key": "case_contrast",
                    "description": "Comparer avec un cas passé et noter les écarts structurants",
                    "cost_est": 0.25,
                    "expected_information_gain": 0.52,
                    "base_gain": 0.27,
                    "keywords": ["analogie", "similaire"],
                }
            ],
        }

    def _extract_features(self, lowered_prompt: str, hypothesis: Hypothesis, strategy: str) -> Dict[str, float]:
        return {
            "bias": 1.0,
            "prompt_len": min(1.0, len(lowered_prompt) / 400.0),
            "contains_question": 1.0
            if any(token in lowered_prompt for token in ["?", "pourquoi", "why", "comment", "how"])
            else 0.0,
            "contains_test": 1.0
            if any(token in lowered_prompt for token in ["test", "verifier", "vérifier", "validation"])
            else 0.0,
            "strategy_match": 1.0 if self._strategy_keyword_match(lowered_prompt, strategy) else 0.0,
            "hypothesis_prior": min(1.0, max(0.0, hypothesis.prior)),
        }

    def _strategy_keyword_match(self, lowered_prompt: str, strategy: str) -> bool:
        keywords = {
            "abduction": ["cause", "pourquoi", "root"],
            "deduction": ["plan", "étapes", "steps", "procedure"],
            "analogy": ["comme", "similaire", "analogie", "exemple"],
        }
        return any(word in lowered_prompt for word in keywords.get(strategy, []))

    def _normalize_preferences(self, prefs: Dict[str, float]) -> None:
        total = sum(prefs.values())
        if total > 0:
            for key in prefs:
                prefs[key] = float(prefs[key] / total)
