from __future__ import annotations
import json, os, re, time, random
from typing import Dict, Any, List

class LiveLexicon:
    def __init__(self, path: str = "data/lexicon.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.lex: Dict[str, Any] = {
            "collocations": {},  # phrase -> {freq:int, liked:int, last:ts}
            "synonyms": {},      # key -> [variants]
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.lex.update(json.load(f))
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.lex, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_from_text(self, text: str, liked: bool = False):
        # extrait des n-grammes simples (2-5 mots) – heuristique légère
        words = re.findall(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?", text)
        for n in (2,3,4,5):
            for i in range(len(words)-n+1):
                phrase = " ".join(words[i:i+n]).lower()
                if len(phrase) < 8: 
                    continue
                c = self.lex["collocations"].setdefault(phrase, {"freq":0, "liked":0, "last":0})
                c["freq"] += 1
                c["last"] = time.time()
                if liked:
                    c["liked"] += 1

    def prefer(self, phrase: str):
        c = self.lex["collocations"].setdefault(phrase.lower(), {"freq":0, "liked":0, "last":0})
        c["liked"] += 1
        c["last"] = time.time()
        self.save()

    def sample_variant(self, key: str, default: str) -> str:
        # renvoie un synonyme/variation si dispo
        variants = self.lex["synonyms"].get(key, [])
        if not variants:
            return default
        return random.choice([default] + variants)

    def sample_collocation(self, novelty: float = 0.3) -> str | None:
        # favorise ce qui est liked ; dose nouveauté
        items = list(self.lex["collocations"].items())
        if not items:
            return None
        # score = liked*2 + log(freq) - pénalité fraicheur si nouveauté haute
        scored = []
        for phrase, meta in items:
            score = meta["liked"]*2 + (1 + meta["freq"]**0.5)
            if novelty > 0.5:
                score -= 0.3 * (time.time() - meta["last"]) / (3600*24 + 1)
            scored.append((score, phrase))
        scored.sort(reverse=True)
        top = [p for _, p in scored[:10]]
        return random.choice(top) if top else None
