import re
import math
import hashlib
from typing import List
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Zàâäéèêëîïôöùûüç0-9]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return _TOKEN_RE.findall(text)

def _hash_to_dim(token: str, dim: int, seed: int = 13) -> int:
    h = hashlib.sha1((str(seed) + token).encode("utf-8")).hexdigest()
    return int(h, 16) % dim

def l2_normalize(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]

def cosine(a: List[float], b: List[float]) -> float:
    # assume both L2-normalized
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class TinyEncoder:
    """
    Encodeur ultraléger par hashing TF (+ heuristique idf locale).
    - aucun paquet externe requis
    - dim par défaut: 256
    - L2 normalisé
    """

    def __init__(self, dim: int = 256, seed: int = 13):
        self.dim = dim
        self.seed = seed

    def encode(self, text: str) -> List[float]:
        toks = tokenize(text)
        if not toks:
            return [0.0] * self.dim
        counts = Counter(toks)
        vec = [0.0] * self.dim
        # TF * pseudo-IDF (idf≈log(1+1/len(token)) comme heuristique)
        for t, tf in counts.items():
            idx = _hash_to_dim(t, self.dim, self.seed)
            idf = math.log(1.0 + 1.0 / max(1, len(t)))
            vec[idx] += tf * (1.0 + idf)
        return l2_normalize(vec)
