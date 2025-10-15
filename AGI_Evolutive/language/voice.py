from __future__ import annotations
import json, os, time, math, re
from typing import Dict, Any, List

class VoiceProfile:
    """
    Profil de voix persistant, influencé par:
    - self_model.persona (tone, values)
    - feedback utilisateur (praise/complaints)
    - lectures aimées (inbox marquées liked)
    - succès/échec de formulation (si tu loggues des retours)
    """
    def __init__(self, self_model, path: str = "data/voice_profile.json"):
        self.self_model = self_model
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.state: Dict[str, Any] = {
            "style": {           # curseurs 0..1
                "formality": 0.4,
                "warmth": 0.6,
                "humor": 0.2,
                "emoji": 0.2,
                "directness": 0.6,
                "analytical": 0.6,
                "storytelling": 0.3,
                "conciseness": 0.6,
            },
            "register_blacklist": [],   # tics à éviter
            "register_whitelist": [],   # tournures à favoriser
            "liked_sources": [],        # inbox:<path>
            "last_update": time.time(),
        }
        self._load()
        self._init_from_persona()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state.update(json.load(f))
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _init_from_persona(self):
        p = (getattr(self.self_model, "state", {}) or {}).get("persona", {})
        tone = (p.get("tone") or "").lower()
        vals = [v.lower() for v in p.get("values", [])]
        # mapping simple (peut être enrichi)
        if "inquisitive" in tone or "analytical" in tone:
            self.state["style"]["analytical"] = min(1.0, self.state["style"]["analytical"] + 0.2)
            self.state["style"]["directness"] = min(1.0, self.state["style"]["directness"] + 0.1)
        if "friendly" in tone or "warm" in tone or "helpful" in tone:
            self.state["style"]["warmth"] = min(1.0, self.state["style"]["warmth"] + 0.2)
            self.state["style"]["emoji"] = min(1.0, self.state["style"]["emoji"] + 0.1)
        if "formal" in tone:
            self.state["style"]["formality"] = min(1.0, self.state["style"]["formality"] + 0.2)
            self.state["style"]["emoji"] = max(0.0, self.state["style"]["emoji"] - 0.1)
        if "precision" in vals:
            self.state["style"]["conciseness"] = min(1.0, self.state["style"]["conciseness"] + 0.1)

    # ---------- Apprentissage ----------
    def update_from_feedback(self, feedback_text: str, positive: bool):
        """Hochets: 'parfait', 'merci', 'trop long', 'trop froid', 'détaille plus', etc."""
        t = feedback_text.lower()
        d = self.state["style"]
        delta = 0.07 if positive else -0.07
        if "trop long" in t or "trop détaillé" in t:
            d["conciseness"] = min(1.0, d["conciseness"] + 0.10 if positive else d["conciseness"] + 0.0)
        if "plus de détails" in t or "argumente" in t:
            d["analytical"] = min(1.0, d["analytical"] + (0.10 if positive else 0.0))
        if "trop froid" in t or "plus chaleureux" in t:
            d["warmth"] = min(1.0, d["warmth"] + 0.12)
        if "trop familier" in t or "plus pro" in t:
            d["formality"] = min(1.0, d["formality"] + 0.12)
            d["emoji"] = max(0.0, d["emoji"] - 0.10)
        if "évite" in t:
            # extrait un mot après "évite"
            m = re.search(r"évite\s+([A-Za-zÀ-ÿ\-]+)", t)
            if m:
                w = m.group(1).lower()
                if w not in self.state["register_blacklist"]:
                    self.state["register_blacklist"].append(w)
        self.state["last_update"] = time.time()
        self.save()

    def update_from_liked_source(self, inbox_path: str, phrases: List[str] | None = None):
        if inbox_path not in self.state["liked_sources"]:
            self.state["liked_sources"].append(inbox_path)
        if phrases:
            for p in phrases:
                if p not in self.state["register_whitelist"]:
                    self.state["register_whitelist"].append(p)
        self.save()

    # getter rapide (pour le renderer)
    def style(self) -> Dict[str, float]:
        return dict(self.state.get("style", {}))
