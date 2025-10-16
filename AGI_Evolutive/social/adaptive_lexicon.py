# AGI_Evolutive/social/adaptive_lexicon.py
# Lexique adaptatif des marqueurs (positifs/négatifs), avec rétention 2-couches :
# - Couche ACTIVE : postérieurs Beta avec décroissance douce (priorise l'actualité)
# - Couche ARCHIVE : totaux stables (sans decay), jamais oubliés
# Réactivation automatique d'anciens marqueurs quand ils réapparaissent.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import re, json, os, time, math, unicodedata

# ----------------- utilitaires -----------------
def _now(): return time.time()
def clamp(x,a=0.0,b=1.0): return max(a, min(b, x))

_STOPWORDS = set("""
le la les un une des de du au aux et ou mais donc car que qui quoi dont où
je tu il elle on nous vous ils elles ne pas plus moins très trop ce cette ces
mon ton son ma ta sa mes tes ses est es suis êtes sont c'est ça ok d' l'
""".split())

_EMOJI_RE = re.compile(
    "["                       # basic emoji ranges
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+", flags=re.UNICODE)

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = re.sub(r"[^\w\s'"+"]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _ngrams(tokens: List[str], nmin=1, nmax=3):
    for n in range(nmin, nmax+1):
        for i in range(len(tokens)-n+1):
            yield " ".join(tokens[i:i+n])

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
    out: List[str] = []
    for t in parts:
        if len(t) <= 1 and t not in {"❤️","👍","👌","👏","🔥","🤣","😂","😅","😆","😍"}:
            continue
        out.append(t)
    return out

# ----------------- entrées -----------------
@dataclass
class LexEntry:
    phrase: str
    # Beta posteriors (couche ACTIVE)
    alpha_pos: float = 1.0
    beta_pos : float = 1.0
    alpha_neg: float = 1.0
    beta_neg : float = 1.0
    uses: int = 0
    last_ts: float = field(default_factory=_now)
    # Per-user léger
    per_user: Dict[str, Dict[str, float]] = field(default_factory=dict) # {user_id: {"pos":a, "neg":b, "uses":n, "last":ts}}
    tags: List[str] = field(default_factory=list)
    # --- ARCHIVE (stables, sans decay) ---
    total_pos: int = 0
    total_neg: int = 0
    first_seen_ts: float = field(default_factory=_now)
    dormant: bool = False

    def p_pos(self) -> float:
        return self.alpha_pos / (self.alpha_pos + self.beta_pos)

    def p_neg(self) -> float:
        return self.alpha_neg / (self.alpha_neg + self.beta_neg)

# ----------------- archive (structure simple) -----------------
@dataclass
class ArchiveEntry:
    phrase: str
    total_pos: int = 0
    total_neg: int = 0
    uses: int = 0
    first_seen_ts: float = field(default_factory=_now)
    last_seen_ts: float = field(default_factory=_now)

# ----------------- lexique adaptatif -----------------
class AdaptiveLexicon:
    """
    Lexique adaptatif global + par utilisateur.
    - observe_message(...) : apprend depuis les n-grams & emojis avec reward multi-source
    - top_markers(...) : retourne les meilleurs marqueurs ACTIFS (dormants exclus)
    - match(...) : détecte la présence d’un marqueur appris (actif ou dormant) dans un message
    Rétention 2-couches : ACTIVE (decay doux) + ARCHIVE (sans decay, réactivation).
    """

    def __init__(self, arch, path: str = "data/lexicon.json", cfg: Optional[Dict[str,Any]] = None):
        self.arch = arch
        self.path = getattr(arch, "lexicon_path", path)
        self.cfg = cfg or self._default_cfg(getattr(arch, "social_critic_cfg_path", None))
        self.archive_path = self.cfg["lexicon_retention"].get("archive_path", "data/lexicon_archive.json")

        self.entries: Dict[str, LexEntry] = {}
        self.archive: Dict[str, ArchiveEntry] = {}
        self._load_active()
        self._load_archive()

        # seeds : injecte POS/NEG statiques comme priors (sans figer)
        seeds = getattr(arch, "lexicon_seeds", None)
        if seeds and isinstance(seeds, dict):
            for p in seeds.get("pos", []):
                e = self._ensure_active(p)
                e.alpha_pos += 2.0
            for p in seeds.get("neg", []):
                e = self._ensure_active(p)
                e.alpha_neg += 2.0
        # housekeeping dormant au chargement
        self._refresh_dormant_flags()

    # ------------- config par défaut -------------
    def _default_cfg(self, critic_cfg_path: Optional[str]) -> Dict[str, Any]:
        # essaie de lire social_critic_config.json si présent
        if critic_cfg_path and os.path.exists(critic_cfg_path):
            try:
                data = json.load(open(critic_cfg_path, "r", encoding="utf-8")) or {}
                if "lexicon_retention" in data:
                    return {"lexicon_retention": data["lexicon_retention"]}
            except Exception:
                pass
        # défauts raisonnables
        return {"lexicon_retention": {
            "decay": 0.995,                 # douce priorisation
            "floor_alpha_beta": 1.0,        # plancher (rien n'est effacé)
            "dormant_after_days": 60,       # au-delà → dormant (si pas revu)
            "revive_boost": 0.4,            # boost à la réactivation
            "archive_path": "data/lexicon_archive.json"
        }}

    # ------------- I/O -------------
    def _load_active(self):
        try:
            if os.path.exists(self.path):
                raw = json.load(open(self.path, "r", encoding="utf-8")) or {}
                for phrase, d in raw.items():
                    # compat: anciennes versions n’ont pas tous les champs
                    self.entries[phrase] = LexEntry(phrase=phrase, **{k:v for k,v in d.items() if k != "phrase"})
        except Exception:
            self.entries = {}

    def _load_archive(self):
        try:
            if os.path.exists(self.archive_path):
                raw = json.load(open(self.archive_path, "r", encoding="utf-8")) or {}
                for phrase, d in raw.items():
                    self.archive[phrase] = ArchiveEntry(phrase=phrase, **{k:v for k,v in d.items() if k != "phrase"})
        except Exception:
            self.archive = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            data = {k: asdict(v) for k, v in self.entries.items()}
            json.dump(data, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_archive(self):
        try:
            os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
            data = {k: asdict(v) for k, v in self.archive.items()}
            json.dump(data, open(self.archive_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------- helpers -------------
    def _ensure_active(self, phrase: str) -> LexEntry:
        if phrase not in self.entries:
            self.entries[phrase] = LexEntry(phrase=phrase)
        return self.entries[phrase]

    def _ensure_archive(self, phrase: str) -> ArchiveEntry:
        if phrase not in self.archive:
            self.archive[phrase] = ArchiveEntry(phrase=phrase)
        return self.archive[phrase]

    def _refresh_dormant_flags(self):
        days = float(self.cfg["lexicon_retention"].get("dormant_after_days", 60))
        if days <= 0: 
            return
        horizon = days * 86400.0
        now = _now()
        for e in self.entries.values():
            e.dormant = (now - float(e.last_ts)) > horizon

    def _apply_decay_and_floor(self, e: LexEntry):
        decay = float(self.cfg["lexicon_retention"].get("decay", 0.995))
        floor = float(self.cfg["lexicon_retention"].get("floor_alpha_beta", 1.0))
        e.alpha_pos = floor + (e.alpha_pos - floor) * decay
        e.beta_pos  = floor + (e.beta_pos  - floor) * decay
        e.alpha_neg = floor + (e.alpha_neg - floor) * decay
        e.beta_neg  = floor + (e.beta_neg  - floor) * decay

    def _reactivate_if_resurfaced(self, e: LexEntry):
        if not e.dormant:
            return
        boost = float(self.cfg["lexicon_retention"].get("revive_boost", 0.4))
        # re-sème la couche ACTIVE à partir des totaux d’archive
        e.alpha_pos = max(e.alpha_pos, 1.0 + boost * float(e.total_pos))
        e.alpha_neg = max(e.alpha_neg, 1.0 + 0.2  * float(e.total_neg))
        e.beta_pos  = max(e.beta_pos,  1.0)
        e.beta_neg  = max(e.beta_neg,  1.0)
        e.dormant = False

    # ------------- API principale -------------
    def observe_message(self, user_msg: str, reward01: float, confidence: float = 0.5, user_id: Optional[str]=None):
        """
        Observe un message utilisateur et met à jour:
        - Couche ACTIVE : posteriors Beta (avec decay & plancher)
        - Couche ARCHIVE : totaux (sans decay), last_seen_ts
        - Réactivation si le marqueur était dormant
        """
        s = _normalize(user_msg or "")
        if not s:
            return
        toks = _tokenize(s)
        grams = list(_ngrams(toks, 1, 3))
        # évite unigrams stopwords
        grams = [g for g in grams if not (len(g.split())==1 and g in _STOPWORDS)]

        now = _now()
        for g in grams:
            e = self._ensure_active(g)
            a = self._ensure_archive(g)

            # réactivation éventuelle si dormant
            self._reactivate_if_resurfaced(e)

            # decay doux + plancher
            self._apply_decay_and_floor(e)

            # update ACTIVE selon reward
            r = clamp(float(reward01), 0.0, 1.0)
            conf = clamp(float(confidence), 0.0, 1.0)

            if r >= 0.6:
                e.alpha_pos += conf
                e.total_pos += 1  # ARCHIVE: totaux stables
            elif r <= 0.4:
                e.beta_pos  += conf
                e.alpha_neg += conf * 0.6
                e.total_neg += 1  # ARCHIVE
            else:
                # neutre: micro stabilisation côté "neg" pour éviter sur-confiance
                e.beta_neg  += conf * 0.1

            e.uses += 1
            e.last_ts = now

            # per-user
            if user_id:
                u = e.per_user.setdefault(user_id, {"pos":1.0,"neg":1.0,"uses":0,"last":now})
                if r >= 0.6: u["pos"] += conf
                elif r <= 0.4: u["neg"] += conf
                u["uses"] += 1; u["last"] = now

            # ARCHIVE : totaux & timestamps sans decay
            a.uses += 1
            a.last_seen_ts = now
            if a.first_seen_ts <= 0: a.first_seen_ts = now

        # maj dormant flags (si longue inactivité sur d'autres entrées)
        self._refresh_dormant_flags()
        self.save()
        self.save_archive()

    def top_markers(self, polarity: str = "pos", k: int = 20, user_id: Optional[str] = None) -> List[str]:
        """
        Classement pour usage ACTIF (réutilisation) — on EXCLUT les dormants.
        """
        scored: List[Tuple[float, str]] = []
        for phrase, e in self.entries.items():
            if e.dormant:
                continue  # on n'encourage pas un dormant
            if polarity == "pos":
                p = e.p_pos()
                if user_id and user_id in e.per_user:
                    u = e.per_user[user_id]
                    bonus = (u["pos"] / (u["pos"] + u["neg"])) - 0.5
                    p += 0.15 * bonus
            else:
                p = e.p_neg()
            if e.uses < 3:
                continue
            scored.append((float(p), phrase))
        scored.sort(reverse=True)
        return [ph for _, ph in scored[:k]]

    def match(self, user_msg: str, polarity: str = "pos", user_id: Optional[str]=None) -> bool:
        """
        Détection PASSIVE (pour le Social Critic) — on doit VOIR un marqueur même s'il est dormant.
        => on ignore le flag dormant ici (ne pas rater un "vieux tic" qui revient).
        """
        s = _normalize(user_msg or "")
        if not s:
            return False
        # set des phrases connues (actives + dormantes). L’archive garde les anciennes aussi.
        phrases = set(self.entries.keys()) | set(self.archive.keys())
        if not phrases:
            return False
        # recherche simple par substring (rapide, robuste aux emojis)
        return any(ph in s for ph in phrases)
