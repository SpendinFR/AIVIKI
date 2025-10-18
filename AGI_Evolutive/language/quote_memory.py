from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import time, re, math, random, os
from . import DATA_DIR, _json_load, _json_save


@dataclass
class Quote:
    text: str
    source: str           # "inbox:path#L42" | "chat:<msg_id>" | "assistant:<ts>"
    ts: float
    liked: bool = False
    uses: int = 0
    last_used: float = 0.0
    lang: str = "fr"
    tags: List[str] = field(default_factory=list)


class QuoteMemory:
    def __init__(self, storage: str = None, max_items: int = 1200):
        self.storage = storage or os.path.join(DATA_DIR, "quotes.json")
        self.max = max_items
        self.items: List[Quote] = []
        self._last_served: Optional[int] = None
        self.load()

    # ---------- Persistence ----------
    def load(self):
        data = _json_load(self.storage, [])
        self.items = [Quote(**q) for q in data if isinstance(q, dict) and "text" in q]
        self.items = self.items[-self.max:]

    def save(self):
        data = [asdict(q) for q in self.items[-self.max:]]
        _json_save(self.storage, data)

    # ---------- Core ----------
    def ingest(self, text: str, source: str, liked: bool=False, tags: Optional[List[str]]=None):
        text = (text or "").strip()
        if not text or len(text.split()) < 3 or len(text) > 180:
            return
        if re.search(r"https?://", text):  # évite bouts de liens
            return
        q = Quote(text=text, source=source, ts=time.time(), liked=liked, tags=(tags or []))
        self.items.append(q)
        if len(self.items) > self.max:
            self.items = self.items[-self.max:]

    def _tok(self, s: str):
        return set(re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", s.lower()))

    def _sim(self, a: str, b: str) -> float:
        A,B = self._tok(a), self._tok(b)
        return len(A & B) / max(1, len(A | B))

    def sample(self, context: str) -> Optional[str]:
        now = time.time()
        best, pool = [], []
        for idx, q in enumerate(self.items):
            recency  = math.exp(-(now - q.ts)/(3*24*3600))         # demi-vie ~2-3j
            overlap  = self._sim(context, q.text)
            liked    = 0.20 if q.liked else 0.0
            cooldown = 0.0 if now - q.last_used > 3*3600 else -0.6
            fatigue  = -0.06*q.uses
            score    = 0.45*recency + 0.35*overlap + liked + cooldown + fatigue
            best.append((score, idx))
        best.sort(reverse=True, key=lambda x: x[0])
        pool = [idx for s,idx in best[:12] if s > 0.25]
        if not pool:
            self._last_served = None
            return None
        pick = random.choice(pool)
        self.items[pick].uses += 1
        self.items[pick].last_used = now
        self._last_served = pick
        return self.items[pick].text

    def reward_last(self, r: float):
        if self._last_served is None:
            return
        q = self.items[self._last_served]
        if r > 0:
            q.liked = True
            q.uses = max(0, q.uses - 1)
        elif r < 0:
            q.uses += 1

    # util: ingestion de fichier texte en extraits "réutilisables"
    def ingest_file_units(self, path: str, liked: bool=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    t = (line or "").strip()
                    if 8 <= len(t.split()) <= 24:
                        self.ingest(t, source=f"inbox:{path}#L{i+1}", liked=liked)
        except Exception:
            pass
