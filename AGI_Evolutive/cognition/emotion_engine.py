import os, json, time
from typing import Dict, Any

class EmotionEngine:
    """
    Humeur/état émotionnel global persistant, influence les drives (homeostasis).
    """
    def __init__(self, path: str = "data/emotions.json"):
        self.path = path
        self.state = {
            "joy": 0.5, "fear": 0.3, "interest": 0.6, "frustration": 0.3,
            "last_update": 0.0
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def update_from_recent_memories(self, recent: list):
        # heuristiques simples
        for m in recent:
            txt = (m.get("text") or "").lower()
            kind = (m.get("kind") or "").lower()
            if any(w in txt for w in ["bravo","merci","good","bien"]): self.state["joy"] = min(1.0, self.state["joy"]+0.03)
            if "error" in kind or any(w in txt for w in ["erreur","fail","mauvais"]): self.state["frustration"] = min(1.0, self.state["frustration"]+0.04)
            if "?" in (m.get("text") or ""): self.state["interest"] = min(1.0, self.state["interest"]+0.02)
        # léger retour au centre
        for k in ("joy","fear","interest","frustration"):
            self.state[k] = 0.98*self.state[k] + 0.02*0.5
        self.state["last_update"] = time.time()
        self._save()

    def modulate_homeostasis(self, homeostasis):
        # influence curiosity/competence/social_bonding
        h = homeostasis.state["drives"]
        h["curiosity"] = max(0, min(1, h["curiosity"] + 0.05*(self.state["interest"]-0.5)))
        h["competence"] = max(0, min(1, h["competence"] + 0.04*(0.6 - self.state["frustration"])))
        h["social_bonding"] = max(0, min(1, h["social_bonding"] + 0.04*(self.state["joy"]-0.5)))
        homeostasis._save()
