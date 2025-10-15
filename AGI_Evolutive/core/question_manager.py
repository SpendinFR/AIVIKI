
# core/question_manager.py
"""Gestionnaire de questions proactives / active learning."""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class QuestionManager:
    """Centralise la génération de questions à forte valeur informationnelle.

    Le gestionnaire maintient une *question bank* priorisée à partir
    d'incertitudes observées, garde en mémoire les questions déjà posées
    récemment et propose un petit lot de questions courtes et univoques.
    """

    QUESTION_LIBRARY: Dict[str, Iterable[str]] = {
        "goal_focus": (
            "Quel est l'objectif prioritaire que tu veux atteindre ?",
            "Sur quelle mission dois-je concentrer mon attention en premier ?",
        ),
        "constraint": (
            "Quelle est la contrainte non négociable à respecter ?",
            "Y a-t-il une échéance précise à ne pas manquer ?",
        ),
        "evidence": (
            "As-tu un exemple ou un document de référence à me partager ?",
            "Peux-tu citer un cas concret pour illustrer ton attente ?",
        ),
        "success_metric": (
            "Quel indicateur me dira que le résultat est satisfaisant ?",
        ),
        "intent_confirmation": (
            "Est-ce que cet objectif reste bien d'actualité pour toi ?",
        ),
    }

    def __init__(self, arch):
        self.arch = arch
        self.pending_questions: List[Dict[str, Any]] = []
        self.question_bank: List[Dict[str, Any]] = []
        self.uncertainty_log: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.asked_recently: Dict[str, float] = {}
        self.question_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.max_queue = 10
        self.last_generated = 0.0
        self.cooldown = 8.0  # secondes minimum entre générations
        self.reask_cooldown = 3600.0

    # ------------------------------------------------------------------
    # Public API
    def add_question(
        self,
        text: str,
        qtype: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0.6,
    ) -> None:
        """Push une question directement dans la banque."""

        if not text:
            return

        entry = {
            "id": metadata.get("id") if metadata else str(uuid.uuid4()),
            "type": qtype,
            "text": text.strip(),
            "score": _clip(priority),
            "meta": metadata or {},
        }
        self._push_to_bank(entry)
        if entry["text"] and not any(entry["text"] == q.get("text") for q in self.pending_questions):
            if len(self.pending_questions) >= self.max_queue:
                self.pending_questions.pop(0)
            self.pending_questions.append(entry)

    def record_information_need(
        self,
        topic: str,
        severity: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        explicit_question: Optional[str] = None,
    ) -> None:
        """Enregistre une incertitude observée pour enrichir la question bank."""

        ts = time.time()
        entry = {
            "ts": ts,
            "topic": topic,
            "severity": _clip(severity),
            "metadata": metadata or {},
        }
        self.uncertainty_log.append(entry)

        question_text = (explicit_question or "").strip()
        if not question_text:
            question_text = self._design_question(topic, metadata)
        if not question_text:
            return
        payload = {
            "id": entry["metadata"].get("id") if entry["metadata"] else str(uuid.uuid4()),
            "type": entry["metadata"].get("type", topic),
            "text": question_text,
            "score": _clip(0.4 + 0.6 * entry["severity"]),
            "meta": entry["metadata"],
        }
        self._push_to_bank(payload)

    def maybe_generate_questions(self) -> None:
        """Sélectionne un petit lot de questions à proposer."""

        now = time.time()
        if now - self.last_generated < self.cooldown:
            return

        self._refresh_from_models()
        self._purge_stale_bank(now)
        ranked = sorted(self.question_bank, key=lambda x: x.get("score", 0.0), reverse=True)

        for candidate in ranked:
            text = candidate.get("text", "")
            if not text:
                continue
            last_asked = self.asked_recently.get(text, 0.0)
            if now - last_asked < self.reask_cooldown:
                continue
            if any(text == q.get("text") for q in self.pending_questions):
                continue
            self.pending_questions.append(candidate)
            if len(self.pending_questions) >= self.max_queue:
                break

        if self.pending_questions:
            self.last_generated = now

    def pop_questions(self) -> List[Dict[str, Any]]:
        """Retourne les questions planifiées et trace l'historique."""

        now = time.time()
        out = []
        while self.pending_questions:
            q = self.pending_questions.pop(0)
            text = q.get("text")
            if not text:
                continue
            self.asked_recently[text] = now
            self.question_history.append({"ts": now, "question": text, "meta": q.get("meta", {})})
            out.append(q)
        return out

    # ------------------------------------------------------------------
    # Internals
    def _refresh_from_models(self) -> None:
        """Alimente la question bank depuis les modules connexes."""

        # Intent model uncertainties → confirmation questions
        intent_model = getattr(self.arch, "intent_model", None)
        if intent_model and hasattr(intent_model, "pending_clarifications"):
            for item in intent_model.pending_clarifications():
                question, priority, meta = item
                self._push_to_bank(
                    {
                        "id": meta.get("id", str(uuid.uuid4())),
                        "type": meta.get("type", "intent_confirmation"),
                        "text": question,
                        "score": _clip(priority),
                        "meta": meta,
                    }
                )

        # Metacognitive uncertainty → clarifier objectif / contraintes
        meta = getattr(self.arch, "metacognition", None)
        try:
            awareness = 1.0 - float(meta.metacognitive_states.get("awareness_level", 0.5))
        except Exception:
            awareness = 0.5
        if awareness > 0.45:
            self.record_information_need("goal_focus", awareness, metadata={"source": "metacog"})

    def _push_to_bank(self, payload: Dict[str, Any]) -> None:
        text = payload.get("text", "").strip()
        if not text:
            return

        existing = next((q for q in self.question_bank if q.get("text") == text), None)
        if existing:
            existing["score"] = max(existing.get("score", 0.0), payload.get("score", 0.0))
            existing.setdefault("meta", {}).update(payload.get("meta", {}))
            return

        self.question_bank.append(payload)
        if len(self.question_bank) > 50:
            # prune the lowest scored entries
            self.question_bank = sorted(self.question_bank, key=lambda x: x.get("score", 0.0), reverse=True)[:50]

    def _design_question(self, topic: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        library = self.QUESTION_LIBRARY.get(topic)
        if not library:
            return None

        # Choisir la formulation la moins utilisée récemment
        history = {entry["question"] for entry in self.question_history}
        for candidate in library:
            if candidate not in history:
                return candidate
        return next(iter(library), None)

    def _purge_stale_bank(self, now: float) -> None:
        freshness_horizon = 8 * 3600  # 8h
        filtered: List[Dict[str, Any]] = []
        for item in self.question_bank:
            meta = item.get("meta") or {}
            ts = meta.get("ts") or 0.0
            if ts and now - float(ts) > freshness_horizon:
                continue
            filtered.append(item)
        self.question_bank = filtered
