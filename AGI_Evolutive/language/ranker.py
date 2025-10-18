# language/ranker.py
import math
import os
from typing import Dict, Any
from . import DATA_DIR, _json_load, _json_save


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class RankerModel:
    """
    Ranker linéaire léger (logit) + apprentissage pairwise online.
    On featurise (contexte, candidate) et on ajuste w via descente sur perte BT-LR.
    """

    def __init__(self, storage=None):
        self.storage = storage or os.path.join(DATA_DIR, "ranker.json")
        self.w: Dict[str, float] = {}
        self.load()

    # --------- Persistence ----------
    def load(self):
        data = _json_load(self.storage, {})
        self.w = data.get("w", {})

    def save(self):
        _json_save(self.storage, {"w": self.w})

    # --------- Features ----------
    def featurize(self, context: Dict[str, Any], text: str) -> Dict[str, float]:
        # Features simples mais robustes
        f: Dict[str, float] = {}
        L = len(text)
        words = text.split()
        sents = max(1, text.count(".") + text.count("!") + text.count("?"))
        avg_len = L / max(1, len(words))
        # basiques
        f["len_chars/1k"] = min(2.0, L / 1000.0)
        f["len_words/100"] = min(2.0, len(words) / 100.0)
        f["avg_token_len"] = min(1.5, avg_len / 7.0)
        f["sent_density"] = min(1.5, sents / max(1, len(words) / 18))

        # “risques” simples
        f["has_allcaps"] = 1.0 if any(w.isupper() and len(w) >= 3 for w in words) else 0.0
        f["too_many_..."] = min(1.5, text.count("...") / 3.0)

        # diversité lexicale approx
        uniq = len(set(w.lower().strip(",.;:!?") for w in words))
        f["type_token"] = min(1.5, uniq / max(1, len(words)))

        # adéquation préférences (si présentes dans context)
        style = context.get("style", {})
        f["pref_warmth"] = float(style.get("warmth", 0.5))
        f["pref_directness"] = float(style.get("directness", 0.5))
        f["pref_hedging"] = float(style.get("hedging", 0.5))

        # pénalités quote longue (tag dans context)
        f["has_quote"] = 1.0 if context.get("has_quote") else 0.0
        f["quote_len/100"] = min(1.5, context.get("quote_len", 0) / 100.0)

        return f

    def _dot(self, f: Dict[str, float]) -> float:
        # auto-init poids à 0
        s = 0.0
        for k, v in f.items():
            s += v * self.w.get(k, 0.0)
        return s

    def score(self, context: Dict[str, Any], text: str) -> float:
        f = self.featurize(context, text)
        return sigmoid(self._dot(f))

    def update_pair(self, context: Dict[str, Any], winner: str, loser: str, lr: float = 0.2):
        fw = self.featurize(context, winner)
        fl = self.featurize(context, loser)
        # prob préférée(w) > préférée(l)
        s_w = self._dot(fw)
        s_l = self._dot(fl)
        # gradient approché (logit pairwise)
        p = sigmoid(s_w - s_l)
        err = 1.0 - p  # on veut p→1
        # mise à jour
        for k in set(list(fw.keys()) + list(fl.keys())):
            gw = fw.get(k, 0.0)
            gl = fl.get(k, 0.0)
            self.w[k] = self.w.get(k, 0.0) + lr * err * (gw - gl)
        # régule doucement
        for k in list(self.w.keys()):
            self.w[k] *= 0.999
