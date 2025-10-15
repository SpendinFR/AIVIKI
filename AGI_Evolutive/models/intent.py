from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Intent:
    label: str
    description: str
    horizon: str = "mid"
    confidence: float = 0.6
    last_seen: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def score(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        freshness = max(0.2, 1.0 - max(0.0, now - self.last_seen) / (7 * 24 * 3600))
        return max(0.0, min(1.0, self.confidence * freshness))


class IntentModel:
    """Modélise les objectifs récurrents exprimés par l'utilisateur."""

    INTENT_PATTERNS: Tuple[Tuple[str, float], ...] = (
        (r"mon objectif(?: principal| prioritaire)? est (.+)", 0.8),
        (r"je veux ([^.?!]+)", 0.7),
        (r"je voudrais ([^.?!]+)", 0.65),
        (r"ma priorité (?:actuelle|principale) est (.+)", 0.85),
        (r"sur le long terme,? je (?:veux|voudrais) (.+)", 0.75),
    )

    def __init__(self, path: str = "data/intent_model.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._intents: Dict[str, Intent] = {}
        self._history: List[Dict[str, float]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        for item in raw.get("intents", []):
            try:
                intent = Intent(
                    label=item["label"],
                    description=item.get("description", item["label"]),
                    horizon=item.get("horizon", "mid"),
                    confidence=float(item.get("confidence", 0.6)),
                    last_seen=float(item.get("last_seen", time.time())),
                    evidence=list(item.get("evidence", [])),
                    tags=list(item.get("tags", [])),
                )
                self._intents[intent.label] = intent
            except Exception:
                continue
        self._history = list(raw.get("history", []))[-200:]

    def save(self) -> None:
        payload = {
            "intents": [
                {
                    "label": intent.label,
                    "description": intent.description,
                    "horizon": intent.horizon,
                    "confidence": intent.confidence,
                    "last_seen": intent.last_seen,
                    "evidence": intent.evidence,
                    "tags": intent.tags,
                }
                for intent in self._intents.values()
            ],
            "history": self._history[-200:],
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Update cycle
    def observe_user_message(self, text: str, *, source: str = "dialogue") -> List[Intent]:
        """Analyse un message pour détecter des intentions explicites."""

        findings: List[Intent] = []
        if not text:
            return findings
        low = text.lower()
        for pattern, prior in self.INTENT_PATTERNS:
            match = re.search(pattern, low)
            if not match:
                continue
            description = match.group(1).strip().rstrip(".,")
            label = self._normalize_label(description)
            intent = self._intents.get(label) or Intent(label=label, description=description)
            intent.confidence = max(intent.confidence, prior)
            intent.last_seen = time.time()
            if source not in intent.tags:
                intent.tags.append(source)
            snippet = text.strip()
            if snippet and snippet not in intent.evidence:
                intent.evidence.append(snippet)
                intent.evidence = intent.evidence[-5:]
            self._intents[label] = intent
            findings.append(intent)
        if findings:
            self.save()
        return findings

    def decay(self, factor: float = 0.97) -> None:
        now = time.time()
        for intent in self._intents.values():
            intent.confidence *= factor
            intent.confidence = max(0.1, min(1.0, intent.confidence))
            if now - intent.last_seen > 14 * 24 * 3600:
                intent.confidence *= 0.8
        self.save()

    def reinforce(self, label: str, delta: float = 0.1) -> None:
        intent = self._intents.get(label)
        if not intent:
            return
        intent.confidence = max(0.0, min(1.0, intent.confidence + delta))
        intent.last_seen = time.time()
        self.save()

    # ------------------------------------------------------------------
    # Queries
    def as_constraints(self, top_k: int = 3) -> List[Dict[str, str]]:
        now = time.time()
        ranked = sorted(self._intents.values(), key=lambda it: it.score(now), reverse=True)
        constraints = []
        for intent in ranked[:top_k]:
            constraints.append(
                {
                    "label": intent.label,
                    "description": intent.description,
                    "horizon": intent.horizon,
                    "confidence": f"{intent.confidence:.2f}",
                }
            )
        return constraints

    def pending_clarifications(self, threshold: float = 0.55) -> List[Tuple[str, float, Dict[str, str]]]:
        """Retourne les questions prioritaires pour confirmer les intentions."""

        now = time.time()
        prompts: List[Tuple[str, float, Dict[str, str]]] = []
        for intent in self._intents.values():
            score = intent.score(now)
            if score >= threshold:
                continue
            question = f"Est-ce que {intent.description} reste une priorité importante ?"
            prompts.append((question, max(0.4, 1.0 - score), {"id": intent.label, "type": "intent_confirmation"}))
        return prompts

    def describe(self) -> Dict[str, Dict[str, float]]:
        now = time.time()
        return {intent.label: {"score": intent.score(now)} for intent in self._intents.values()}

    # ------------------------------------------------------------------
    # Helpers
    @staticmethod
    def _normalize_label(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug or f"intent-{int(time.time())}"
