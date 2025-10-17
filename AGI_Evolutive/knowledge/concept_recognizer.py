from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re, time, json, os, unicodedata

# ---------- utils ----------
def _now(): return time.time()
def clamp(x,a=0,b=1): return max(a, min(b, x))

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _clean_term(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

STOP = set("""
le la les un une des de du au aux et ou mais donc car que qui quoi dont où
je tu il elle on nous vous ils elles ne pas plus moins très trop ce cette ces
mon ton son ma ta sa mes tes ses est es suis êtes sont c'est ça d' l'
""".split())

SUF_N = ("ie","té","tion","sion","isme","ence","ance","eur","ure","ment","esse","isme")
SUF_A = ("ique","if","ive","el","elle","al","ale","aire","eux","euse")
SUF_V = ("iser","ifier","iser","iser","ifier","er","ir","re")  # repères faibles

# ---------- patrons génériques (déf, reformulation, étiquette, citations…) ----------
RE_DEF_1 = re.compile(r"\b([A-Za-zÀ-ÿ][\w\- ]{2,})\b\s+(?:est|sont|c['e]st|signifie|désigne|correspond(?:\s+à)?|implique|se manifeste(?:\s+par)?)\s", re.I)
RE_DEF_2 = re.compile(r"(?:qu['e]st-ce que|c['e]st quoi)\s+([A-Za-zÀ-ÿ][\w\- ]{2,})\b\??", re.I)
RE_LABEL = re.compile(r"^([A-Za-zÀ-ÿ][\w\- ]{2,})\s*:\s+.+$", re.M)
RE_QUOTE = re.compile(r"[«\"]([A-Za-zÀ-ÿ][\w\- ]{2,})[»\"]")
RE_REFORM = re.compile(r"\b(autrement dit|en d'autres termes|en clair|pour dire simple|c'est-à-dire)\b", re.I)
RE_RHET_Q = re.compile(r"\?\s*$")

# ---------- indices de style ----------
RE_IRONY = re.compile(r"\b(si tu le dis|bien sûr+|ouais c'est ça|lol+|mdr+|ptdr+)\b", re.I)
RE_FORMAL = re.compile(r"\b(cependant|toutefois|néanmoins|en revanche|par conséquent|de surcroît)\b", re.I)
RE_SLANG = re.compile(r"\b(chelou|wesh|grave|nan|ouais|bg|relou|bcp|tmtc)\b", re.I)
RE_EMOJIS = re.compile(r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]+")

def _is_stopish(t: str) -> bool:
    toks = [w for w in t.split() if w]
    if not toks: return True
    if len(toks) == 1 and toks[0] in STOP: return True
    return False

def _is_concepty(t: str) -> bool:
    # "conceptuel" par morpho/longueur/structure
    if _is_stopish(t): return False
    if len(t) < 3: return False
    if any(t.endswith(s) for s in SUF_N): return True
    if " " in t and len(t.split()) <= 4: return True
    if any(t.endswith(s) for s in SUF_A): return True
    return True  # garde ouvert

@dataclass
class ItemCandidate:
    # kind: "concept" | "term" | "style" | "construction"
    kind: str
    label: str
    score: float
    evidence: Dict[str, Any]
    features: Dict[str, Any]
    ts: float

# ---------- ConceptRecognizer ----------
class ConceptRecognizer:
    """
    Détecte des candidats "à apprendre" dans un texte. Hybride:
    - patrons génériques (définitions/étiquettes/citations/reformulations/questions)
    - morphologie FR (suffixes nom/adjectif/verbe)
    - OOV/termes fréquents inconnus (skills/ontology)
    - indices dialogiques (via miner: acts -> hint concept)
    - styles et constructions
    Apprend (renforce) ses propres patrons quand un item est confirmé.
    """
    def __init__(self, arch, patterns_path: str = "data/concept_patterns.json"):
        self.arch = arch
        self.path = getattr(arch, "concept_patterns_path", patterns_path)
        self.patterns: Dict[str, Dict[str, Any]] = self._load_patterns()

    # --- patterns dynamiques : poids par famille ---
    def _load_patterns(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                return json.load(open(self.path, "r", encoding="utf-8")) or {}
        except Exception:
            pass
        return {
            # familles -> poids init (évoluent)
            "def_1": {"w": 0.45},       # "X est / c'est / signifie ..."
            "def_2": {"w": 0.40},       # "qu'est-ce que X ? / c'est quoi X ?"
            "label": {"w": 0.30},       # "X : ..."
            "quote": {"w": 0.25},       # «X»
            "reform": {"w": 0.18},      # "autrement dit", etc. -> construction
            "morph": {"w": 0.22},       # suffixes conceptuels
            "verb":  {"w": 0.15},       # suffixes verbaux (faible)
            "oov":   {"w": 0.25},       # inconnu des skills/ontology (freq≥2)
            "dialog": {"w":0.20},       # indice via actes (ex: empathy_act)
            # styles
            "style_irony":  {"w": 0.40},
            "style_formal": {"w": 0.35},
            "style_slang":  {"w": 0.35},
            "style_emoji":  {"w": 0.25},
            # constructions (questions rhétoriques, etc.)
            "construction_rhetq": {"w": 0.22},
        }

    def save_patterns(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump(self.patterns, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- extraction principale ---
    def extract_candidates(self, text: str, dialog_hints: Optional[Dict[str, Any]] = None) -> List[ItemCandidate]:
        textN = _norm(text)
        evC: Dict[str, Dict[str, Any]] = {}   # evidence par label
        scores: Dict[Tuple[str,str], float] = {}  # (kind,label) -> score
        feats: Dict[Tuple[str,str], Dict[str, Any]] = {}

        def bump(kind: str, label: str, family: str, amount: float, evidence: Any):
            k = (kind, label)
            scores[k] = scores.get(k, 0.0) + amount
            evC.setdefault(label, {}).setdefault(family, [])
            evC[label][family].append(evidence)
            feats.setdefault(k, {"families": set()})
            feats[k]["families"].add(family)

        # A) Concepts par patrons définitionnels / labels / citations
        for m in RE_DEF_1.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "def_1", self.patterns["def_1"]["w"], m.group(0))
        for m in RE_DEF_2.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "def_2", self.patterns["def_2"]["w"], m.group(0))
        for m in RE_LABEL.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "label", self.patterns["label"]["w"], m.group(0))
        for m in RE_QUOTE.finditer(textN):
            t = _clean_term(m.group(1))
            if _is_concepty(t):
                bump("concept", t, "quote", self.patterns["quote"]["w"], m.group(0))

        # B) Constructions (reformulation, c-à-d)
        for m in RE_REFORM.finditer(textN):
            bump("construction", "reformulation", "reform", self.patterns["reform"]["w"], m.group(0))

        # C) Styles
        if RE_IRONY.search(textN):
            bump("style", "ironie", "style_irony", self.patterns["style_irony"]["w"], "irony_marker")
        if RE_FORMAL.search(textN):
            bump("style", "formel", "style_formal", self.patterns["style_formal"]["w"], "formal_markers")
        if RE_SLANG.search(textN):
            bump("style", "argot", "style_slang", self.patterns["style_slang"]["w"], "slang_markers")
        if RE_EMOJIS.search(textN):
            bump("style", "emoji_usage", "style_emoji", self.patterns["style_emoji"]["w"], "emoji_present")

        # D) Morphologie (concepts & verbes candidats)
        for w in re.findall(r"[A-Za-zÀ-ÿ][\w\-]{2,}", textN):
            t = _clean_term(w)
            if not t or t in STOP: 
                continue
            if any(t.endswith(s) for s in SUF_N) or any(t.endswith(s) for s in SUF_A):
                bump("concept", t, "morph", self.patterns["morph"]["w"], t)
            if any(t.endswith(s) for s in SUF_V):
                bump("term", t, "verb", self.patterns["verb"]["w"], t)

        # E) OOV / fréquences -> termes à apprendre (même 1 mot)
        known = set()
        try:
            # skills connus
            skills_path = getattr(self.arch, "skills_path", "data/skills.json")
            if os.path.exists(skills_path):
                data = json.load(open(skills_path, "r", encoding="utf-8")) or {}
                known |= set(data.keys())
        except Exception: pass
        try:
            onto = getattr(self.arch, "ontology", None)
            if onto and hasattr(onto, "all_concepts"):
                known |= set(onto.all_concepts())
        except Exception: pass

        freqs: Dict[str,int] = {}
        for w in re.findall(r"[A-Za-zÀ-ÿ][\w\-]{2,}", textN):
            t = _clean_term(w)
            if not _is_stopish(t):
                freqs[t] = freqs.get(t,0)+1

        for t, f in freqs.items():
            if t not in known and f >= 2:
                # "term" (mot/locution) candidat à apprendre
                bump("term", t, "oov", self.patterns["oov"]["w"] * min(1.0, (f-1)/3.0), {"freq":f})

        # F) Indices dialogiques (hints passés par le miner)
        # Exemple: {"empathy_act": True} → augmente le poids du concept "empathie"
        if dialog_hints and dialog_hints.get("empathy_act"):
            bump("concept", "empathie", "dialog", self.patterns["dialog"]["w"], "empathy_act")

        # G) Questions rhétoriques (construction)
        for line in re.split(r"[\r\n]+", textN):
            if RE_RHET_Q.search(line.strip()):
                bump("construction", "question_rhetorique", "construction_rhetq", self.patterns["construction_rhetq"]["w"], line.strip())

        # Assemblage final
        items: List[ItemCandidate] = []
        for (kind, label), s in scores.items():
            label_clean = _clean_term(label)
            if not label_clean:
                continue
            # concepty pour "concept", souple pour "term"
            if kind == "concept" and not _is_concepty(label_clean):
                continue
            # score borné
            s = clamp(s, 0.0, 1.0)
            evidence = evC.get(label, {})
            fams = feats.get((kind,label), {}).get("families", set())
            items.append(ItemCandidate(kind=kind, label=label_clean, score=s,
                                       evidence=evidence, features={"families": sorted(list(fams))}, ts=_now()))
        # tri: score puis richesse de l'evidence
        items.sort(key=lambda c: (c.score, sum(len(v) if isinstance(v,list) else 1 for v in c.evidence.values())), reverse=True)
        return items

    # --- apprentissage des patrons à partir d'une confirmation ---
    def learn_from_confirmation(self, kind: str, label: str, evidence: Dict[str, Any], reward: float = 0.8):
        # Renforce doucement les familles présentes dans l'evidence
        families = list(evidence.keys() or [])
        for fam in families:
            if fam in self.patterns:
                w = float(self.patterns[fam].get("w", 0.2))
                self.patterns[fam]["w"] = clamp(w + 0.05 * reward, 0.05, 0.8)
        self.save_patterns()

    def learn_from_rejection(self, kind: str, label: str, evidence: Dict[str, Any], penalty: float = 0.5):
        for fam in list(evidence.keys() or []):
            if fam in self.patterns:
                w = float(self.patterns[fam].get("w", 0.2))
                self.patterns[fam]["w"] = max(0.05, w - 0.05 * penalty)
        self.save_patterns()

    # --- helpers d'intégration (faciles à appeler) ---
    def commit_candidates_to_memory(self, source: str, items: List[ItemCandidate], arch=None):
        arch = arch or self.arch
        for it in items:
            arch.memory.add_memory({
                "kind": f"{it.kind}_candidate",
                "label": it.label,
                "score": it.score,
                "evidence": it.evidence,
                "features": it.features,
                "ts": it.ts,
                "source": source
            })

    def autogoals_for_high_confidence(self, items: List[ItemCandidate], arch=None,
                                      th_concept: float = 0.72, th_term: float = 0.75,
                                      th_style: float = 0.80, th_constr: float = 0.70):
        """
        Crée des goals non préemptifs selon le type :
        - concept -> learn_concept::<label>
        - term    -> learn_term::<label> (ou learn_concept si tu n'as pas d'action 'learn_term')
        - style   -> learn_style::<label> (ajustement de voix)
        - construction -> learn_construction::<label>
        """
        arch = arch or self.arch
        for it in items:
            if it.kind == "concept" and it.score >= th_concept and not self._known(arch, it.label):
                gid = f"learn_concept::{it.label}"
                arch.planner.ensure_goal(gid, f"Apprendre le concept « {it.label} »", priority=0.72, tags=["background"])
                arch.planner.add_action_step(gid, "learn_concept", {"concept": it.label}, priority=0.70)
            elif it.kind == "term" and it.score >= th_term and not self._known(arch, it.label):
                # si tu n'as pas d'action 'learn_term', retombe sur learn_concept
                action = "learn_term" if hasattr(getattr(arch, "io", None), "_h_learn_term") else "learn_concept"
                gid = f"{action}::{it.label}"
                arch.planner.ensure_goal(gid, f"Apprendre le terme « {it.label} »", priority=0.68, tags=["background"])
                arch.planner.add_action_step(gid, action, {"term": it.label}, priority=0.66)
            elif it.kind == "style" and it.score >= th_style:
                gid = f"learn_style::{it.label}"
                arch.planner.ensure_goal(gid, f"Comprendre le style « {it.label} »", priority=0.60, tags=["style"])
                arch.planner.add_action_step(gid, "learn_style", {"style": it.label, "evidence": it.evidence}, priority=0.58)
            elif it.kind == "construction" and it.score >= th_constr:
                gid = f"learn_construction::{it.label}"
                arch.planner.ensure_goal(gid, f"Étudier la construction « {it.label} »", priority=0.60, tags=["linguistics"])
                arch.planner.add_action_step(gid, "learn_construction", {"construction": it.label}, priority=0.58)

    def _known(self, arch, label: str) -> bool:
        # connu si présent en skills/ontology (tu peux étendre)
        try:
            skills_path = getattr(arch, "skills_path", "data/skills.json")
            if os.path.exists(skills_path):
                data = json.load(open(skills_path, "r", encoding="utf-8")) or {}
                if label in data: return True
        except Exception:
            pass
        try:
            onto = getattr(arch, "ontology", None)
            if onto and hasattr(onto, "has_concept"):
                return bool(onto.has_concept(label))
        except Exception:
            pass
        return False
