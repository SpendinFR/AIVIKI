# Lexique adaptatif des marqueurs (positifs/négatifs) — apprend depuis les échanges.
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import re, json, os, time
import unicodedata


def _now() -> float:
    return time.time()


_STOPWORDS = set(
    """
le la les un une des de du au aux et ou mais donc car que qui quoi dont où
je tu il elle on nous vous ils elles ne pas plus moins très trop ce cette ces
mon ton son ma ta sa mes tes ses est es suis êtes sont c'est ça ok
""".split()
)

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    # keep emojis as tokens, strip other punct except apostrophes
    s = re.sub(r"[^\w\s'"+"]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _ngrams(tokens: List[str], nmin: int = 1, nmax: int = 3):
    for n in range(nmin, nmax + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


def _tokenize(s: str) -> List[str]:
    # split emojis into separate tokens and words
    em_split = _EMOJI_RE.split(s)
    emojis = _EMOJI_RE.findall(s)
    parts: List[str] = []
    for i, chunk in enumerate(em_split):
        words = [w for w in re.split(r"\s+", chunk) if w]
        parts += words
        if i < len(emojis):
            parts.append(emojis[i])
    # remove stopwords for unigrams only
    out: List[str] = []
    for t in parts:
        if len(t) <= 1 and t not in {"❤️", "👍", "👌", "👏", "🔥", "🤣", "😂", "😅", "😆", "😍"}:
            continue
        out.append(t)
    return out


@dataclass
class LexEntry:
    phrase: str
    # Beta posteriors sur polarité
    alpha_pos: float = 1.0
    beta_pos: float = 1.0
    alpha_neg: float = 1.0
    beta_neg: float = 1.0
    uses: int = 0
    last_ts: float = field(default_factory=_now)
    # per-user reinforcement (light)
    per_user: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def p_pos(self) -> float:
        return self.alpha_pos / (self.alpha_pos + self.beta_pos)

    def p_neg(self) -> float:
        return self.alpha_neg / (self.alpha_neg + self.beta_neg)

    def observe(
        self,
        r01: float,
        user_id: Optional[str] = None,
        conf: float = 0.5,
        decay: float = 0.995,
    ) -> None:
        # Décroissance douce
        self.alpha_pos = 1 + (self.alpha_pos - 1) * decay
        self.beta_pos = 1 + (self.beta_pos - 1) * decay
        self.alpha_neg = 1 + (self.alpha_neg - 1) * decay
        self.beta_neg = 1 + (self.beta_neg - 1) * decay

        # Update selon reward agrégé [0..1]
        if r01 >= 0.6:
            self.alpha_pos += conf
        elif r01 <= 0.4:
            self.beta_pos += conf
            self.alpha_neg += conf * 0.6
        else:
            # neutre: très léger update pour stabiliser
            self.beta_neg += conf * 0.1

        self.uses += 1
        self.last_ts = _now()

        if user_id:
            u = self.per_user.setdefault(
                user_id, {"pos": 1.0, "neg": 1.0, "uses": 0, "last": _now()}
            )
            if r01 >= 0.6:
                u["pos"] += conf
            elif r01 <= 0.4:
                u["neg"] += conf
            u["uses"] += 1
            u["last"] = _now()


class AdaptiveLexicon:
    """Lexique adaptatif global + par utilisateur."""

    def __init__(self, arch: Any, path: str = "data/lexicon.json") -> None:
        self.arch = arch
        self.path = getattr(arch, "lexicon_path", path)
        self.entries: Dict[str, LexEntry] = {}
        self._load()
        # seeds : injecte les POS/NEG statiques comme priors, sans les figer
        seeds = getattr(arch, "lexicon_seeds", None)
        if seeds and isinstance(seeds, dict):
            for p in seeds.get("pos", []) or []:
                self._ensure(p).alpha_pos += 2.0
            for p in seeds.get("neg", []) or []:
                self._ensure(p).alpha_neg += 2.0

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for phrase, meta in data.items():
                    self.entries[phrase] = LexEntry(phrase=phrase, **meta)
        except Exception:
            self.entries = {}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            data = {k: asdict(v) for k, v in self.entries.items()}
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _ensure(self, phrase: str) -> LexEntry:
        if phrase not in self.entries:
            self.entries[phrase] = LexEntry(phrase=phrase)
        return self.entries[phrase]

    def _maybe_expand_with_ontology(self, entry: LexEntry) -> None:
        if entry.uses < 5 or entry.p_pos() <= 0.75:
            return
        if "onto_seeded" in entry.tags:
            return
        onto = getattr(self.arch, "ontology", None)
        if not onto:
            return
        variants: List[str] = []
        synonyms_fn = getattr(onto, "synonyms", None)
        if callable(synonyms_fn):
            try:
                variants.extend(synonyms_fn(entry.phrase) or [])
            except Exception:
                pass
        if not variants:
            neighbors_fn = getattr(onto, "neighbors", None)
            if callable(neighbors_fn):
                try:
                    neigh = neighbors_fn(entry.phrase)
                    if isinstance(neigh, dict):
                        variants.extend(str(k) for k in neigh.keys())
                    elif isinstance(neigh, (list, tuple, set)):
                        variants.extend(str(v) for v in neigh)
                except Exception:
                    pass
        clean_variants = []
        for var in variants:
            if not isinstance(var, str):
                continue
            norm = _normalize(var)
            if not norm or norm == entry.phrase:
                continue
            clean_variants.append(norm)
        if not clean_variants:
            return
        entry.tags.append("onto_seeded")
        for phrase in clean_variants[:5]:
            seeded = self._ensure(phrase)
            seeded.alpha_pos += 0.5
            seeded.tags = list(set(seeded.tags + ["ontology_seed"]))

    # ------------------------------------------------------------------
    def observe_message(
        self,
        user_msg: str,
        reward01: float,
        confidence: float = 0.5,
        user_id: Optional[str] = None,
    ) -> None:
        s = _normalize(user_msg or "")
        toks = _tokenize(s)
        grams = list(_ngrams(toks, 1, 3))
        # filtre : on évite n-grams dominés par stopwords en unigram
        grams = [g for g in grams if not (len(g.split()) == 1 and g in _STOPWORDS)]
        for g in grams:
            entry = self._ensure(g)
            entry.observe(reward01, user_id=user_id, conf=confidence)
            self._maybe_expand_with_ontology(entry)
        self.save()

    # ------------------------------------------------------------------
    def top_markers(
        self,
        polarity: str = "pos",
        k: int = 20,
        user_id: Optional[str] = None,
    ) -> List[str]:
        scored: List[Tuple[float, str]] = []
        for phrase, entry in self.entries.items():
            if entry.uses < 3:
                continue
            if polarity == "pos":
                score = entry.p_pos()
                if user_id and user_id in entry.per_user:
                    user_meta = entry.per_user[user_id]
                    total = user_meta.get("pos", 1.0) + user_meta.get("neg", 1.0)
                    if total > 0:
                        bonus = (user_meta.get("pos", 1.0) / total) - 0.5
                        score += 0.15 * bonus
            else:
                score = entry.p_neg()
            scored.append((score, phrase))
        scored.sort(reverse=True)
        return [phrase for score, phrase in scored[:k] if score > 0]

    def match(
        self,
        user_msg: str,
        polarity: str = "pos",
        user_id: Optional[str] = None,
    ) -> bool:
        s = _normalize(user_msg or "")
        if not s:
            return False
        markers = set(self.top_markers(polarity=polarity, k=50, user_id=user_id))
        if not markers:
            return False
        return any(m in s for m in markers)
