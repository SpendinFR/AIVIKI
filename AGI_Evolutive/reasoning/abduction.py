from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import datetime as dt

@dataclass
class Hypothesis:
    label: str
    explanation: str
    score: float
    priors: Dict[str, float]
    evidence: List[str]
    ask_next: Optional[str] = None

class AbductiveReasoner:
    """
    Pipeline:
      - generators -> candidats
      - scoring -> score(H) par priors + matches + beliefs
      - active question (ask_next) si tie
      - calibration (log prédiction) gérée côté arch
    """
    def __init__(self, beliefs, user_model, generators: Optional[List[Callable[[str], List[Tuple[str,str]]]]] = None):
        self.beliefs = beliefs
        self.user = user_model
        self.generators = generators or [self._g_default]

    def _g_default(self, text: str) -> List[Tuple[str, str]]:
        t = text.lower()
        if "jaune" in t and ("ongle" in t or "doigt" in t):
            return [
                ("cheddar", "résidu de fromage fondu"),
                ("curcuma", "résidu d'épice"),
                ("nicotine", "tache nicotine"),
                ("infection", "sécrétion ou mycose"),
                ("autre", "autre cause"),
            ]
        # fallback neutre
        return [("inconnu", "cause inconnue faute d'indices")]

    def _time(self) -> dt.datetime:
        return dt.datetime.now()

    def _score(self, label: str, text: str) -> Tuple[float, Dict[str,float], List[str]]:
        t = text.lower(); now = self._time()
        pri = 0.5
        ev: List[str] = []
        # priors utilisateur
        pri = max(pri, self.user.prior(label)) if hasattr(self.user, "prior") else pri
        pri = min(1.0, pri + self.user.routine_bias(label, now)) if hasattr(self.user, "routine_bias") else pri
        # beliefs
        boost = 0.0
        for b in self.beliefs.query(subject="user", relation="likes", min_conf=0.6):
            if b.value in {label, f"{label}_related"}:
                boost += 0.15
        # matches (features toy; à étendre par domaine)
        matches = 0.0
        if "midi" in t: matches += 0.05
        if "réchauff" in t or "micro" in t: matches += 0.05
        if label in t: matches += 0.2
        score = max(0.0, min(1.0, 0.5*pri + boost + matches))
        return score, {"pri": pri, "boost": boost, "matches": matches}, ev

    def generate(self, observation_text: str) -> List[Hypothesis]:
        cands: List[Tuple[str,str]] = []
        for g in self.generators:
            try: cands.extend(g(observation_text))
            except Exception: continue
        # dedup
        uniq = {}
        for lab, why in cands:
            uniq.setdefault(lab, why)
        hyps: List[Hypothesis] = []
        for lab, why in uniq.items():
            s, pri, ev = self._score(lab, observation_text)
            hyps.append(Hypothesis(label=lab, explanation=why, score=s, priors=pri, evidence=ev))
        hyps.sort(key=lambda h: h.score, reverse=True)
        # active question si nécessaire
        if len(hyps) >= 2 and (hyps[0].score - hyps[1].score) < 0.12:
            hyps[0].ask_next = f"Question pour départager « {hyps[0].label} » et « {hyps[1].label} » : {self._craft_question(hyps[0].label, hyps[1].label)}"
        return hyps[:5]

    def _craft_question(self, a: str, b: str) -> str:
        # squelette générique
        return f"As-tu un indice concret qui pointerait plutôt vers {a} (trace, odeur, contexte horaire) ou {b} ?"
