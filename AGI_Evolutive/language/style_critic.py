from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U0001F1E6-\U0001F1FF]"
)
HEDGING_RE = re.compile(
    r"\b(p(?:e|é|è|ê)ut[-\s]?(?:etre|être))\b",
    flags=re.IGNORECASE,
)
DOUBLE_ADVERB_RE = re.compile(r"\b(tr[eéè]s)\s+(tr[eéè]s)\b", flags=re.IGNORECASE)
COPULA_RE = re.compile(
    r"\b(?:c['’]?est|est)\s+(?:un|une|le|la|l['’])",
    flags=re.IGNORECASE,
)
PUNCT_BEFORE_RE = re.compile(r"\s+([!?;:])")
PUNCT_AFTER_RE = re.compile(r"([!?;:])(?!\s)")


def _strip_accents(text: str) -> str:
    """Return a lowercase, accent-less representation of *text*."""

    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        ch.lower()
        for ch in normalized
        if not unicodedata.combining(ch)
    )


@dataclass
class SignalSnapshot:
    """Expose the internal state of an adaptive signal."""

    observation: float
    pressure: float
    momentum: float
    baseline: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "observation": self.observation,
            "pressure": self.pressure,
            "momentum": self.momentum,
            "baseline": self.baseline,
        }


@dataclass
class AdaptiveSignal:
    """Streaming signal with a touch of evolution via momentum."""

    baseline: float
    sensitivity: float = 1.0
    decay: float = 0.35
    reward_strength: float = 0.3
    floor: float = 0.0
    momentum: float = 0.0

    def observe(self, observation: float) -> SignalSnapshot:
        pressure = max(0.0, observation - self.baseline)
        self.momentum = (1.0 - self.decay) * self.momentum + self.decay * pressure
        amplified = pressure * (1.0 + self.sensitivity * self.momentum)
        return SignalSnapshot(
            observation=observation,
            pressure=amplified,
            momentum=self.momentum,
            baseline=self.baseline,
        )

    def reinforce(self, reward: float) -> None:
        delta = reward * self.reward_strength
        self.baseline = max(self.floor, self.baseline + delta)
        self.momentum *= 0.5


class EvolvingSignalModel:
    """Minimal orchestration for adaptive signals."""

    def __init__(self, signals: Dict[str, AdaptiveSignal]):
        self._signals = signals
        self._snapshots: Dict[str, SignalSnapshot] = {}

    def observe(self, name: str, observation: float) -> SignalSnapshot:
        signal = self._signals[name]
        snapshot = signal.observe(observation)
        self._snapshots[name] = snapshot
        return snapshot

    def reinforce(self, name: str, reward: float) -> None:
        if name in self._signals:
            self._signals[name].reinforce(reward)

    def last_snapshots(self) -> Dict[str, Dict[str, float]]:
        return {name: snapshot.as_dict() for name, snapshot in self._snapshots.items()}


def _expressive_density(text: str) -> float:
    bang_runs = sum(1 for _ in re.finditer(r"!{2,}", text))
    question_runs = sum(1 for _ in re.finditer(r"\?{2,}", text))
    ellipsis = text.count("...")
    emoji = len(EMOJI_RE.findall(text))
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    caps_ratio = 0.0
    if words:
        caps_words = sum(1 for word in words if len(word) > 2 and word.isupper())
        caps_ratio = caps_words / len(words)

    return (
        0.6 * bang_runs
        + 0.45 * question_runs
        + 0.3 * ellipsis
        + 0.4 * emoji
        + 1.5 * caps_ratio
    )


class StyleCritic:
    """Critique légère du style pour post-traiter les réponses."""

    def __init__(self, max_chars: int = 1200, signal_overrides: Dict[str, AdaptiveSignal] | None = None):
        self.max_chars = int(max_chars)
        defaults: Dict[str, AdaptiveSignal] = {
            "too_long": AdaptiveSignal(baseline=float(self.max_chars), sensitivity=0.02, decay=0.4),
            "excess_bang": AdaptiveSignal(baseline=0.4, sensitivity=1.2, decay=0.3),
            "hedging_maybe": AdaptiveSignal(baseline=0.1, sensitivity=1.5, decay=0.35),
            "adverb_dup": AdaptiveSignal(baseline=0.15, sensitivity=1.4, decay=0.3),
            "copula_definition": AdaptiveSignal(baseline=0.4, sensitivity=1.1, decay=0.3),
            "expressive_noise": AdaptiveSignal(baseline=0.8, sensitivity=1.0, decay=0.25),
        }
        if signal_overrides:
            defaults.update(signal_overrides)
        self._signals = EvolvingSignalModel(defaults)

    def analyze(self, text: str) -> Dict[str, Any]:
        sample = (text or "").strip()
        normalized = _strip_accents(sample)
        issues: List[Tuple[str, Any]] = []

        if len(sample) > self.max_chars:
            snapshot = self._signals.observe("too_long", float(len(sample)))
            issues.append(("too_long", round(snapshot.pressure, 3)))

        double_bang_matches = list(re.finditer(r"!{2,}", sample))
        if double_bang_matches:
            snapshot = self._signals.observe("excess_bang", float(len(double_bang_matches)))
            if snapshot.pressure > 0.0:
                issues.append(("excess_bang", round(snapshot.pressure, 3)))

        double_adverbs = DOUBLE_ADVERB_RE.findall(normalized)
        if double_adverbs:
            snapshot = self._signals.observe("adverb_dup", float(len(double_adverbs)))
            if snapshot.pressure > 0.0:
                issues.append(("adverb_dup", {"severity": round(snapshot.pressure, 3), "examples": double_adverbs}))

        if HEDGING_RE.search(normalized):
            snapshot = self._signals.observe("hedging_maybe", 1.0)
            if snapshot.pressure > 0.0:
                issues.append(("hedging_maybe", round(snapshot.pressure, 3)))

        copula_hits = list(COPULA_RE.finditer(sample))
        if copula_hits:
            snapshot = self._signals.observe("copula_definition", float(len(copula_hits)))
            if snapshot.pressure > 0.0:
                issues.append(("copula_definition", round(snapshot.pressure, 3)))

        expressive = _expressive_density(sample)
        if expressive:
            snapshot = self._signals.observe("expressive_noise", expressive)
            if snapshot.pressure > 0.0:
                issues.append(("expressive_noise", round(snapshot.pressure, 3)))

        return {
            "length": len(sample),
            "issues": issues,
            "signals": self._signals.last_snapshots(),
        }

    def rewrite(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)

        def _replace_probablement(match: re.Match[str]) -> str:
            original = match.group(0)
            replacement = "probablement"
            return replacement.capitalize() if original[:1].isupper() else replacement

        cleaned = HEDGING_RE.sub(_replace_probablement, cleaned)
        cleaned = DOUBLE_ADVERB_RE.sub(lambda m: m.group(1), cleaned)
        cleaned = re.sub(r"([!?]){2,}", r"\1", cleaned)
        cleaned = PUNCT_BEFORE_RE.sub(r"\1", cleaned)
        cleaned = PUNCT_AFTER_RE.sub(r"\1 ", cleaned)
        cleaned = cleaned.strip()

        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip()
            if not cleaned.endswith((".", "!", "?", "…")):
                cleaned = cleaned.rstrip("…") + "…"

        return cleaned

    def nudge(self, issue_name: str, reward: float) -> None:
        """Integrate external feedback to keep the critic evolving."""

        self._signals.reinforce(issue_name, reward)
