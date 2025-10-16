from __future__ import annotations

from typing import Dict, Any, List
import re
import time


class StyleObserver:
    """Observe les textes lus et ajuste la voix / le lexique en douceur."""

    def __init__(self, self_model, homeostasis, voice_profile, lexicon, user_model=None) -> None:
        self.self_model = self_model
        self.homeo = homeostasis
        self.voice = voice_profile
        self.lex = lexicon
        self.user_model = user_model
        self.last_updates: List[float] = []  # timestamps pour rate limit

    # --------- API publique ----------
    def observe_text(self, text: str, source: str, *, channel: str) -> None:
        """Analyse un texte et déclenche des micro-ajustements si pertinent."""
        if not text or len(text) < 20:
            return

        # 1) extraire candidates (collocations, refs, blagues courtes)
        items = self._extract_candidates(text)

        # 2) scorer l'appétence (0..1)
        persona = self._persona_profile()
        drives = (getattr(self.homeo, "state", {}) or {}).get("drives", {})
        scored: List[tuple[float, Dict[str, Any]]] = []
        for it in items:
            score = self._like_score(it, persona, drives, channel=channel)
            if score >= 0.65 and self._aligned(it, persona):
                scored.append((score, it))

        if not scored:
            return

        # 3) rate limit (max 5 ajouts / 5min)
        now = time.time()
        self.last_updates = [t for t in self.last_updates if now - t < 300]
        budget = max(0, 5 - len(self.last_updates))
        if budget <= 0:
            return

        scored.sort(reverse=True)
        picked = [it for _, it in scored[:budget]]

        # 4) appliquer (petits incréments)
        for it in picked:
            self._apply_like(it)
            self.last_updates.append(now)

    # --------- Extracteurs ----------
    def _extract_candidates(self, text: str) -> List[Dict[str, Any]]:
        # Collocations 2–5 mots + détection simple d’idiomes/blagues/refs
        words = re.findall(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?", text)
        items: List[Dict[str, Any]] = []
        for n in (2, 3, 4, 5):
            for i in range(len(words) - n + 1):
                phr = " ".join(words[i : i + n])
                if len(phr) < 8:
                    continue
                items.append({"type": "collocation", "text": phr})

        # blagues ou punchlines ultra basiques (heuristique)
        if re.search(r"\b(c’est (pas )?faux|plot twist|fun fact)\b", text, flags=re.I):
            items.append({"type": "punch", "text": "fun fact"})
        # références pop (à étendre)
        if re.search(r"\b(matrix|inception|star wars|one piece)\b", text, flags=re.I):
            items.append({"type": "reference", "text": "réf pop-culture"})

        return items

    # --------- Scoring ----------
    def _like_score(
        self,
        item: Dict[str, Any],
        persona: Dict[str, Any],
        drives: Dict[str, float],
        *,
        channel: str,
    ) -> float:
        tone = (persona.get("tone") or "").lower()
        raw_vals = persona.get("values", [])
        if isinstance(raw_vals, dict):
            vals = [key.lower() for key, level in raw_vals.items() if level]
        else:
            vals = [v.lower() for v in raw_vals]

        # features simples
        txt = item["text"].lower()
        novelty = 1.0 if not self.lex.lex["collocations"].get(txt) else 0.4
        humor = 1.0 if item["type"] in ("punch",) else 0.2
        analytic = 1.0 if len(txt.split()) >= 3 and item["type"] == "collocation" else 0.4

        # match drives (si homeo impulsion “social_bonding”/“curiosity”)
        cur = float(drives.get("curiosity", 0.5))
        bond = float(drives.get("social_bonding", 0.5))

        # alignement persona simple
        align = 0.5
        if "analytical" in tone:
            align += 0.2 * analytic
        if "friendly" in tone or "warm" in tone:
            align += 0.2 * bond
        if "precision" in vals:
            align += 0.1

        # canal : booste l’utilisateur (on apprend plus de toi)
        chan = 0.15 if channel == "user" else 0.0

        raw = 0.35 * align + 0.25 * novelty + 0.2 * humor + 0.2 * cur + chan
        return max(0.0, min(1.0, raw))

    def _aligned(self, item: Dict[str, Any], persona: Dict[str, Any]) -> bool:
        # garde-fou : blacklist mots, pas d’insulte, pas contraire aux valeurs, etc.
        txt = item["text"].lower()
        if len(txt) > 60:
            return False
        # exemples de filtres:
        bad = ["insulte", "raciste"]  # à remplacer par ta vraie liste
        if any(b in txt for b in bad):
            return False
        return True

    # --------- Application ----------
    def _apply_like(self, item: Dict[str, Any]) -> None:
        txt = item["text"].strip()
        # Favoriser la collocation dans le lexique
        try:
            self.lex.prefer(txt)
        except Exception:
            pass
        # Petite adaptation de la voix si pertinent
        try:
            if item["type"] == "punch":
                self.voice.bump("humor", +0.03)
                self.voice.bump("storytelling", +0.02)
            else:
                self.voice.bump("analytical", +0.02)
        except Exception:
            pass
        # Enregistre une mémoire de style appris (pour traçabilité)
        try:
            state = getattr(self.self_model, "state", None)
            if state is not None:
                state.setdefault("style_learned", []).append({"text": txt, "ts": time.time(), "source": "style_observer"})
        except Exception:
            pass

    def _persona_profile(self) -> Dict[str, Any]:
        persona: Dict[str, Any] = {}
        try:
            if self.self_model is not None:
                base_state = getattr(self.self_model, "state", {}) or {}
                persona = dict(base_state.get("persona", {}) or {})
        except Exception:
            persona = {}
        if self.user_model is not None and hasattr(self.user_model, "describe"):
            try:
                user_state = self.user_model.describe() or {}
                user_persona = user_state.get("persona") or {}
                if user_persona:
                    merged = dict(persona)
                    merged.update(user_persona)
                    persona = merged
            except Exception:
                pass
        return persona
