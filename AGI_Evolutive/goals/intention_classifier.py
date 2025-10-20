"""Lightweight online text classifier for goal intentions."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from .dag_store import GoalNode

if TYPE_CHECKING:
    from . import GoalMetadata

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Prediction:
    label: Optional[str]
    confidence: float


class OnlineTextClassifier:
    """Simple multinomial Naïve Bayes classifier updated online."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.label_counts: Counter[str] = Counter()
        self.feature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.total_feature_counts: Counter[str] = Counter()
        self.examples: list[tuple[str, str]] = []  # (text, label)
        self._load()

    # ------------------------------------------------------------------
    def predict(self, text: str) -> Prediction:
        tokens = self._vectorize(text)
        if not self.label_counts:
            return Prediction(label=None, confidence=0.0)

        log_probs: Dict[str, float] = {}
        total_labels = sum(self.label_counts.values())
        vocab_size = len(self.total_feature_counts) or 1

        for label, label_count in self.label_counts.items():
            log_prob = math.log(label_count / total_labels)
            feature_total = sum(self.feature_counts[label].values())
            denom = feature_total + vocab_size
            for token, freq in tokens.items():
                token_count = self.feature_counts[label].get(token, 0)
                log_prob += freq * math.log((token_count + 1) / denom)
            log_probs[label] = log_prob

        max_label, max_log_prob = max(log_probs.items(), key=lambda item: item[1])
        exp_probs = {label: math.exp(lp - max_log_prob) for label, lp in log_probs.items()}
        total_prob = sum(exp_probs.values())
        confidence = exp_probs[max_label] / total_prob if total_prob else 0.0
        return Prediction(label=max_label, confidence=confidence)

    def update(self, label: str, text: str) -> None:
        tokens = self._vectorize(text)
        if not tokens:
            return
        self.label_counts[label] += 1
        self.examples.append((text, label))
        for token, freq in tokens.items():
            self.feature_counts[label][token] += freq
            self.total_feature_counts[token] += freq
        self._persist()

    # ------------------------------------------------------------------
    def _vectorize(self, text: str) -> Counter[str]:
        text = (text or "").lower()
        tokens = TOKEN_PATTERN.findall(text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        counts = Counter(tokens)
        counts.update(bigrams)
        return counts

    def _load(self) -> None:
        if not self.data_path or not os.path.exists(self.data_path):
            return
        try:
            with open(self.data_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return
        label_counts = payload.get("label_counts") or {}
        feature_counts = payload.get("feature_counts") or {}
        total_feature_counts = payload.get("total_feature_counts") or {}
        examples = payload.get("examples") or []

        self.label_counts.update(label_counts)
        for label, counts in feature_counts.items():
            self.feature_counts[label].update(counts)
        self.total_feature_counts.update(total_feature_counts)
        for example in examples[-500:]:
            text = example.get("text", "")
            label = example.get("label")
            if text and label:
                self.examples.append((text, label))

    def _persist(self) -> None:
        if not self.data_path:
            return
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        payload = {
            "label_counts": dict(self.label_counts),
            "feature_counts": {label: dict(counter) for label, counter in self.feature_counts.items()},
            "total_feature_counts": dict(self.total_feature_counts),
            "examples": [
                {"text": text, "label": label}
                for text, label in self.examples[-500:]
            ],
        }
        with open(self.data_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


class IntentionModel:
    """Wraps the classifier with metadata-aware helpers."""

    def __init__(self, data_path: str) -> None:
        self.classifier = OnlineTextClassifier(data_path=data_path)

    def predict(self, goal: GoalNode, metadata: Optional["GoalMetadata"]) -> Prediction:
        payload = self._compose_payload(goal, metadata)
        return self.classifier.predict(payload)

    def update(self, label: str, goal: GoalNode, metadata: Optional["GoalMetadata"]) -> None:
        payload = self._compose_payload(goal, metadata)
        self.classifier.update(label, payload)

    @staticmethod
    def _compose_payload(goal: GoalNode, metadata: Optional["GoalMetadata"]) -> str:
        text_parts = [goal.description or ""]
        if goal.criteria:
            text_parts.extend(goal.criteria)
        if metadata:
            text_parts.append(metadata.goal_type.value)
            text_parts.extend(metadata.success_criteria)
        return " \n ".join(filter(None, text_parts))
