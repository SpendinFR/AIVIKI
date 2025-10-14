import os
import re
import json
import time
import math
import glob
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ---------------- Utils ----------------


def _now() -> float:
    return time.time()


def _safe_json_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_jsonl_append(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------- Extractor ----------------

BASIC_STOP = set(
    """
    a au aux avec c ce ces ça dans de des du elle elles en et eux il ils je la le les leur lui ma mais me même mes
    moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre
    vous y d l n t s qu' j' c' m' n' t' s' auj aujourd'hui puis donc alors comme si car or ni soit été être suis es est
    sont étais étions étiez étaient sera seront serai serons serez serais seraient étais étiez étaient
    """.split()
)

CAP_WORD = re.compile(r"\b[A-ZÉÈÀÂÎÔÙÛÇ][\w\-'']+\b")
TOKEN = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç'']{3,}")


class ConceptExtractor:
    """
    - Balaye les mémoires récentes (ou data/memories/ en fallback)
    - Extrait des concepts (mots clés, bigrammes, entités capitalisées simples)
    - Construit un graphe de co-occurrence (concept_graph.json)
    - Persiste un index (concept_index.json) + jsonl des événements (concepts.jsonl)
    - Enregistre des mémoires 'concept_extracted' optionnellement
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.paths = {
            "concept_index": os.path.join(self.data_dir, "concept_index.json"),
            "concept_graph": os.path.join(self.data_dir, "concept_graph.json"),
            "concept_events": os.path.join(self.data_dir, "concepts.jsonl"),
            "state": os.path.join(self.data_dir, "concept_state.json"),
        }
        self.bound: Dict[str, Optional[Any]] = {
            "memory": None,
            "emotions": None,
            "metacog": None,
            "language": None,
        }

        # état persistant léger
        self.state = _safe_json_load(
            self.paths["state"],
            {
                "last_run": 0.0,
                "processed_ids": [],
                "df": {},  # document frequency par concept
                "total_docs": 0,
            },
        )
        self.concept_index: Dict[str, Dict[str, Any]] = _safe_json_load(
            self.paths["concept_index"],
            {
                # concept -> stats
                # "émotions humaines": {"count": 3, "first_seen": ts, "last_seen": ts, "sources": [1, 5], "sample": "texte"}
            },
        )
        self.graph = _safe_json_load(
            self.paths["concept_graph"],
            {
                # "nodes": {"concept": {"count": int}},
                # "edges": {"concept|||concept2": {"w": float}}
                "nodes": {},
                "edges": {},
            },
        )

        self.period_s = 15.0  # throttling
        self._last_step = 0.0

    # ---------- binding ----------
    def bind(self, memory=None, emotions=None, metacog=None, language=None):
        self.bound.update(
            {
                "memory": memory,
                "emotions": emotions,
                "metacog": metacog,
                "language": language,
            }
        )

    # ---------- stepping ----------
    def step(self, force: bool = False, max_batch: int = 400):
        now = time.time()
        if not force and now - self._last_step < self.period_s:
            return
        self._last_step = now
        self.run_once(max_batch=max_batch)

    # ---------- core ----------
    def run_once(self, max_batch: int = 400):
        mems = self._fetch_recent_memories(limit=max_batch)
        if not mems:
            return

        docs: List[Tuple[Optional[str], str, Dict[str, Any]]] = []
        ids_in_batch: List[str] = []
        for m in mems:
            mid = m.get("id") or m.get("_id") or m.get("memory_id") or None
            if mid and mid in self.state["processed_ids"]:
                continue
            txt = (m.get("content") or "").strip()
            if not txt:
                continue
            docs.append((mid, txt, m))
            if mid:
                ids_in_batch.append(mid)
from typing import List, Dict, Any, Tuple, Optional
import re
import math
import time
from collections import Counter
from .concept_store import ConceptStore

STOPWORDS = set(
    """
    a à ai as au aux avec ça ce ces cet cette d de des du elle elles en et eux il ils je la le les leur leurs lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses si son sur ta te tes toi ton tu un une vos votre vous y
    the of and to a in is it that for on as with this these from by at or an be are was were i you he she they we me him her them my your our their not do does did have has had can could will would should into about over more no so than then very just
    """.split()
)

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")


class ConceptExtractor:
    """Convertit memories -> concepts et relations basées sur la cooccurrence."""

    def __init__(self, store: ConceptStore):
        self.store = store
        self.min_len = 3
        self.window = 5
        self.last_dashboard = 0.0

    def step(self, memory_system, max_batch: int = 300) -> None:
        memories = self._safe_get_recent_memories(memory_system, n=max_batch)
        if not memories:
            return

        docs: List[Tuple[str, str]] = []
        for memory in memories:
            mid = str(memory.get("id") or memory.get("_id") or memory.get("t"))
            text = self._memory_to_text(memory)
            if not mid or not text:
                continue
            docs.append((mid, text))

        if not docs:
            return

        # mise à jour df/total_docs
        for mid, txt, _ in docs:
            uniq_concepts = set(self._extract_concepts(txt))
            for concept in uniq_concepts:
                self.state["df"][concept] = self.state["df"].get(concept, 0) + 1
            self.state["total_docs"] += 1

        # scoring tf-idf lite + graph
        for mid, txt, meta in docs:
            concepts = self._extract_concepts(txt)
            if not concepts:
                continue
            tf = Counter(concepts)
            top = self._score_select(tf, self.state["df"], self.state["total_docs"], k=6)
            self._update_index(top, mid, txt)
            self._update_graph(top)

            # log jsonl
            _safe_jsonl_append(
                self.paths["concept_events"],
                {
                    "t": _now(),
                    "memory_id": mid,
                    "top_concepts": top,
                    "kind": meta.get("kind"),
                    "meta": meta.get("metadata", {}),
                },
            )

            # optionnel : écrire une mémoire structurée
            mem = self.bound.get("memory")
            if mem and hasattr(mem, "add_memory"):
                try:
                    mem.add_memory(
                        kind="concept_extracted",
                        content="; ".join([concept for concept, _ in top]),
                        metadata={"source_memory": mid, "concepts": top},
                    )
                except Exception:
                    pass

        # marquer comme traités
        self.state["processed_ids"] += ids_in_batch
        # compacter de temps en temps
        if len(self.state["processed_ids"]) > 5000:
            self.state["processed_ids"] = self.state["processed_ids"][-2500:]

        self._save_state()

    # ---------- concept extraction ----------
    def _extract_concepts(self, text: str) -> List[str]:
        # entités capitalisées simples (acteurs/noms propres)
        caps = [match.group(0) for match in CAP_WORD.finditer(text)]
        caps_norm = [cap.strip("''").lower() for cap in caps if len(cap) >= 3]

        # tokens (lower)
        lowered = text.lower()
        toks = TOKEN.findall(lowered)
        toks = [token for token in toks if token not in BASIC_STOP and len(token) >= 3]

        # bigrammes légers (collocations)
        bigrams: List[str] = []
        for idx in range(len(toks) - 1):
            w1, w2 = toks[idx], toks[idx + 1]
            if w1 in BASIC_STOP or w2 in BASIC_STOP:
                continue
            bigrams.append(f"{w1} {w2}")

        # pondérer : caps (2x), bigrams (1.5x), tokens (1x)
        concepts = toks + bigrams + caps_norm + caps_norm  # caps doublés
        return concepts

    def _score_select(
        self, tf: Counter, df: Dict[str, int], total_docs: int, k: int = 6
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for term, freq in tf.items():
            idf = math.log(1 + (total_docs / max(1, df.get(term, 1))))
            score = freq * idf
            scored.append((term, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    # ---------- index/graph ----------
    def _update_index(
        self, top: List[Tuple[str, float]], mem_id: Optional[str], text: str
    ) -> None:
        now = _now()
        for concept, score in top:
            entry = self.concept_index.get(
                concept,
                {
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "sources": [],
                    "sample": "",
                },
            )
            entry["count"] += 1
            entry["last_seen"] = now
            if mem_id:
                if len(entry["sources"]) < 8:
                    entry["sources"].append(mem_id)
            if not entry["sample"]:
                # extrait court pour contexte
                entry["sample"] = text[:240]
            self.concept_index[concept] = entry

        # persister l'index des concepts
        with open(self.paths["concept_index"], "w", encoding="utf-8") as f:
            json.dump(self.concept_index, f, ensure_ascii=False, indent=2)

    def _update_graph(self, top: List[Tuple[str, float]]) -> None:
        """Met à jour le graphe de cooccurrence pour les concepts détectés."""

        if not top:
            return

        # mise à jour des noeuds
        for concept, _ in top:
            node = self.graph["nodes"].get(concept, {"count": 0})
            node["count"] += 1
            self.graph["nodes"][concept] = node

        # mise à jour des arêtes (co-occurrences)
        concepts = [concept for concept, _ in top]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                key = f"{concepts[i]}|||{concepts[j]}"
                edge = self.graph["edges"].get(key, {"w": 0.0})
                edge["w"] += 1.0
                self.graph["edges"][key] = edge

        # persister l'état du graphe
        with open(self.paths["concept_graph"], "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False)

    def _save_state(self) -> None:
        self.state["last_run"] = _now()
        with open(self.paths["state"], "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    # ---------- memory fetch ----------
    def _fetch_recent_memories(self, limit: int = 400) -> List[Dict[str, Any]]:
        mem = self.bound.get("memory")
        if mem is None:
            # fallback : lecture de data/memories/*.json
            return self._fallback_read_files(limit=limit)
        # utiliser interfaces si dispo
        try:
            if hasattr(mem, "get_recent_memories"):
                res = mem.get_recent_memories(n=limit) or []
                return res
        except Exception:
            pass
        try:
            if hasattr(mem, "iter_memories"):
                res = []
                for item in mem.iter_memories():
                    res.append(item)
                    if len(res) >= limit:
                        break
                return res
        except Exception:
            pass
        return self._fallback_read_files(limit=limit)

    def _fallback_read_files(self, limit: int = 400) -> List[Dict[str, Any]]:
        root = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(root):
            return []
        files = sorted(glob.glob(os.path.join(root, "*.json")), reverse=True)
        out: List[Dict[str, Any]] = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
        tokens_by_doc = [(mid, self._tokenize(text)) for mid, text in docs]
        df: Counter[str] = Counter()
        for _, tokens in tokens_by_doc:
            df.update(set(tokens))

        co_counts: Counter[Tuple[str, str]] = Counter()
        for _, tokens in tokens_by_doc:
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 1 + self.window, len(tokens))):
                    a, b = tokens[i], tokens[j]
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    co_counts[(a, b)] += 1

        total_docs = max(1, len(tokens_by_doc))
        for term, docfreq in df.items():
            support_delta = docfreq / total_docs
            salience_delta = 0.0
            if any(self._term_in_memory(term, memory) and (memory.get("emotion") or memory.get("reward")) for memory in memories):
                salience_delta += 0.1
            self.store.upsert_concept(
                label=term,
                support_delta=support_delta,
                salience_delta=salience_delta,
                example_mem_id=None,
                confidence=0.6,
            )

        N = total_docs
        for (a, b), count_ab in co_counts.items():
            p_ab = count_ab / N
            p_a = df[a] / N
            p_b = df[b] / N
            if p_ab <= 0 or p_a <= 0 or p_b <= 0:
                continue
            pmi = math.log(p_ab / (p_a * p_b) + 1e-9)
            if pmi <= 0.0:
                continue
            concept_a = self._get_concept_id(a)
            concept_b = self._get_concept_id(b)
            if concept_a and concept_b:
                self.store.upsert_relation(concept_a, concept_b, "related_to", weight_delta=float(pmi), mem_id=None, confidence=0.6)

        now = time.time()
        if now - self.last_dashboard > 15:
            self.store.save()
            self.last_dashboard = now

    def _get_concept_id(self, label: str) -> Optional[str]:
        lower = label.lower()
        for cid, concept in self.store.concepts.items():
            if concept.label.lower() == lower:
                return cid
        return None

    def _safe_get_recent_memories(self, memory_system, n: int = 300) -> List[Dict[str, Any]]:
        try:
            if hasattr(memory_system, "get_recent_memories"):
                return memory_system.get_recent_memories(n=n) or []
            if hasattr(memory_system, "memories"):
                return list(memory_system.memories)[-n:]
        except Exception:
            pass
        return []

    def _memory_to_text(self, memory: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("text", "content", "utterance", "summary", "note", "plan", "solution", "question", "answer"):
            value = memory.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = WORD.findall(text)
        tokens: List[str] = []
        for token in raw_tokens:
            lower = token.lower()
            if len(lower) < self.min_len:
                continue
            if lower in STOPWORDS:
                continue
            tokens.append(lower)
        return tokens

    def _term_in_memory(self, term: str, memory: Dict[str, Any]) -> bool:
        text = self._memory_to_text(memory).lower()
        return term.lower() in text
import os, json, time
from typing import Dict, Any, List
from collections import Counter

class ConceptExtractor:
    """
    Extrait des "concepts" (mots/phrases clés dominants) des souvenirs récents.
    Persiste dans data/concepts.json
    """
    def __init__(self, memory_store, data_path: str = "data/concepts.json"):
        self.memory = memory_store
        self.path = data_path
        self.state = {"concepts": {}, "updated_at": 0.0}
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

    def extract_from_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        recents = self.memory.get_recent_memories(n=n)
        cnt = Counter()
        for m in recents:
            text = (m.get("text") or "").lower()
            for w in text.split():
                if w.isalpha() and len(w) >= 4:
                    cnt[w] += 1
        top = cnt.most_common(20)
        out = []
        for w,fq in top:
            score = min(1.0, 0.2 + fq/20.0)
            cur = self.state["concepts"].get(w, {"count": 0, "score": 0.0})
            cur["count"] += fq
            cur["score"] = max(cur["score"], score)
            self.state["concepts"][w] = cur
            out.append({"concept": w, "score": score})
        self.state["updated_at"] = time.time()
        self._save()
        return out
