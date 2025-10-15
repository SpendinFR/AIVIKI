from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, json, datetime as dt, time

class UserModel:
    """
    - valeurs & persona (déclaratif)
    - préférences (Beta-like: (pos, neg) -> prob)
    - routines temporelles (jour/heure -> activités probables)
    - extraction/learning implicite depuis la mémoire
    """
    def __init__(self, path: str = "data/user_model.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.state: Dict[str, Any] = {
            "persona": {"tone": "neutral", "values": {"honesty": 1.0, "helpfulness": 1.0}},
            "preferences": {},  # label -> {"pos": int, "neg": int, "prob": float}
            "routines": {},     # "Tue:12" -> {"fast_food": 0.6, "gym": 0.1}
            "last_update": time.time(),
        }
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try: self.state = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception: pass

    def save(self) -> None:
        json.dump(self.state, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- Persona / valeurs ----------
    def set_value(self, key: str, level: float) -> None:
        self.state["persona"]["values"][key] = float(max(0.0, min(1.0, level))); self.save()

    def set_tone(self, tone: str) -> None:
        self.state["persona"]["tone"] = tone; self.save()

    def describe(self) -> Dict[str, Any]:
        return self.state

    # ---------- Préférences (apprentissage Beta-like) ----------
    def _ensure_pref(self, label: str) -> Dict[str, Any]:
        p = self.state["preferences"].get(label)
        if not p:
            p = {"pos": 1, "neg": 1, "prob": 0.5}
            self.state["preferences"][label] = p
        return p

    def observe_preference(self, label: str, liked: bool) -> None:
        p = self._ensure_pref(label)
        if liked: p["pos"] += 1
        else:     p["neg"] += 1
        p["prob"] = p["pos"] / float(p["pos"] + p["neg"])
        self.save()

    def prior(self, label: str) -> float:
        p = self._ensure_pref(label); return float(p["prob"])

    # ---------- Routines temporelles ----------
    @staticmethod
    def _key_for_time(t: Optional[dt.datetime] = None) -> str:
        t = t or dt.datetime.now()
        return f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][t.weekday()]}:{t.hour:02d}"

    def observe_routine(self, label: str, t: Optional[dt.datetime] = None, strength: float = 0.2) -> None:
        k = self._key_for_time(t)
        cur = self.state["routines"].get(k, {})
        cur[label] = float(max(0.0, min(1.0, cur.get(label, 0.0) + strength)))
        self.state["routines"][k] = cur; self.save()

    def routine_bias(self, label: str, t: Optional[dt.datetime] = None) -> float:
        k = self._key_for_time(t)
        return float(self.state["routines"].get(k, {}).get(label, 0.0))

    # ---------- Learning implicite depuis la mémoire ----------
    def ingest_memories(self, memories: List[Dict[str, Any]]) -> int:
        """
        Parse des entrées mémoire pour détecter des signaux:
        - 'like:<label>' / 'dislike:<label>'
        - 'did:<activity>'
        - 'tone:<style>'
        """
        n = 0
        for m in memories[-200:]:
            text = (m.get("content") or m.get("text") or "").lower()
            meta = m.get("metadata") or {}
            if not text: continue
            if "like:" in text:
                # ex: "like:burger_king"
                for tok in text.split():
                    if tok.startswith("like:"):
                        self.observe_preference(tok.split(":",1)[1], True); n += 1
            if "dislike:" in text:
                for tok in text.split():
                    if tok.startswith("dislike:"):
                        self.observe_preference(tok.split(":",1)[1], False); n += 1
            if "did:" in text:
                # ex: "did:fast_food"
                for tok in text.split():
                    if tok.startswith("did:"):
                        self.observe_routine(tok.split(":",1)[1]); n += 1
            if "tone:" in text:
                for tok in text.split():
                    if tok.startswith("tone:"):
                        self.set_tone(tok.split(":",1)[1]); n += 1
        return n
