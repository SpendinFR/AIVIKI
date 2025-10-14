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
    vous y d l n t s qu’ j’ c’ m’ n’ t’ s’ auj aujourd’hui puis donc alors comme si car or ni soit été être suis es est
    sont étais étions étiez étaient sera seront serai serons serez serais seraient étais étiez étaient
    """.split()
)

CAP_WORD = re.compile(r"\b[A-ZÉÈÀÂÎÔÙÛÇ][\w\-’']+\b")
TOKEN = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç’']{3,}")


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
                # "émotions humaines": {"count": 3, "first_seen": ts, "last_seen": ts, "sources": [ids...], "sample": "..."}
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
        caps_norm = [cap.strip("’’").lower() for cap in caps if len(cap) >= 3]

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
            # node graph
            node = self.graph["nodes"].get(concept, {"count": 0})
            node["count"] += 1
            self.graph["nodes"][concept] = node

        # edges (co-occur)
        concepts = [concept for concept, _ in top]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                key = f"{concepts[i]}|||{concepts[j]}"
                edge = self.graph["edges"].get(key, {"w": 0.0})
                edge["w"] += 1.0
                self.graph["edges"][key] = edge

        # persister
        with open(self.paths["concept_index"], "w", encoding="utf-8") as f:
            json.dump(self.concept_index, f, ensure_ascii=False, indent=2)
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
        return out
