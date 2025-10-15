from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .concept_store import ConceptStore

BASIC_STOP = set(
    """
    a au aux avec c ce ces ça dans de des du elle elles en et eux il ils je la le les leur lui ma mais me même mes
    moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre
    vous y d l n t s qu' j' c' m' n' t' s' auj aujourd'hui puis donc alors comme si car or ni soit été être suis es est
    sont étais étions étiez étaient sera seront serai serons serez serais seraient étais étiez étaient
    the of and to a in is it that for on as with this these from by at or an be are was were i you he she they we me him her them my your our their
    not do does did have has had can could will would should into about over more no so than then very just
    """.split()
)

CAP_WORD = re.compile(r"\b[A-ZÉÈÀÂÎÔÙÛÇ][\w\-']+\b")
TOKEN = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç']{3,}")


def _now() -> float:
    return time.time()


def _safe_json_load(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _safe_jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _normalize_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        for key in ("content", "text", "summary", "description", "message", "label", "concept", "value"):
            candidate = value.get(key)
            normalized = _normalize_text(candidate)
            if normalized:
                return normalized
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_text(item)
            if normalized:
                return normalized
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _memory_to_text(memory: Dict[str, Any]) -> Optional[str]:
    for key in ("content", "text", "summary", "message", "body"):
        if key in memory:
            normalized = _normalize_text(memory.get(key))
            if normalized:
                return normalized
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        normalized = _normalize_text(metadata)
        if normalized:
            return normalized
    return None


class ConceptExtractor:
    """Concept detection pipeline shared across orchestrator and semantic manager."""

    def __init__(
        self,
        memory_store: Optional[Any],
        data_path: str = "data/concepts.json",
    ) -> None:
        concept_events_path: Optional[str]
        if not data_path:
            self.data_dir = "."
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")
        elif data_path.endswith(".jsonl"):
            base_dir = os.path.dirname(data_path)
            self.data_dir = base_dir if base_dir else "."
            concept_events_path = data_path
        elif data_path.endswith(".json"):
            base_dir = os.path.dirname(data_path)
            self.data_dir = base_dir if base_dir else "."
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")
        else:
            self.data_dir = data_path
            concept_events_path = os.path.join(self.data_dir, "concept_events.jsonl")

        os.makedirs(self.data_dir, exist_ok=True)
        self.paths = {
            "concept_index": os.path.join(self.data_dir, "concept_index.json"),
            "concept_graph": os.path.join(self.data_dir, "concept_graph.json"),
            "concept_events": concept_events_path,
            "state": os.path.join(self.data_dir, "concept_state.json"),
        }
        self.store: Optional[ConceptStore] = None
        self.bound: Dict[str, Optional[Any]] = {
            "memory": memory_store,
            "emotions": None,
            "metacog": None,
            "language": None,
        }

        self.state = _safe_json_load(
            self.paths["state"],
            {
                "last_run": 0.0,
                "processed_ids": [],
                "df": {},
                "total_docs": 0,
            },
        )
        self.concept_index: Dict[str, Dict[str, Any]] = _safe_json_load(
            self.paths["concept_index"],
            {},
        )
        self.graph = _safe_json_load(
            self.paths["concept_graph"],
            {"nodes": {}, "edges": {}},
        )

        self.period_s = 15.0
        self._last_step = 0.0

    # ---------- binding ----------
    def bind(self, memory: Any = None, emotions=None, metacog=None, language=None) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        self.bound.update({"emotions": emotions, "metacog": metacog, "language": language})

    # ---------- scheduling helpers ----------
    def step(self, memory: Any = None, max_batch: int = 300) -> None:
        if memory is not None:
            self.bound["memory"] = memory
        memory_system = memory if memory is not None else self.bound.get("memory")
        if memory_system is None:
            return
        now = time.time()
        if now - self._last_step < self.period_s:
            return
        self._last_step = now
        self._run_batch(self._collect_memories(memory_system, limit=max_batch))

    def run_once(self, max_batch: int = 400) -> None:
        mems = self._collect_memories(self.bound.get("memory"), limit=max_batch)
        self._run_batch(mems)

    def extract_from_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        """Compatibility helper used by the orchestrator scheduler."""
        mems = self._collect_memories(self.bound.get("memory"), limit=n)
        aggregated = self._run_batch(mems)
        result = [
            {"concept": concept, "score": score}
            for concept, score in aggregated
        ]
        return result

    # ---------- core processing ----------
    def _run_batch(self, memories: Iterable[Dict[str, Any]]) -> List[Tuple[str, float]]:
        docs = list(self._prepare_documents(memories))
        if not docs:
            return []

        aggregated: Counter[str] = Counter()
        ids_in_batch: List[str] = []

        for mem_id, text, meta in docs:
            concepts = self._extract_concepts(text)
            if not concepts:
                continue
            tf = Counter(concepts)
            uniq_concepts = set(tf.keys())
            self.state["total_docs"] += 1
            for concept in uniq_concepts:
                self.state["df"][concept] = self.state["df"].get(concept, 0) + 1

            top = self._score_select(tf, self.state["df"], self.state["total_docs"], k=6)
            for concept, score in top:
                aggregated[concept] += score

            self._update_index(top, mem_id, text)
            self._update_graph(top)
            self._log_event(mem_id, meta, top)
            self._update_store(mem_id, top)
            self._log_memory(mem_id, top)

            if mem_id:
                ids_in_batch.append(mem_id)

        self.state["processed_ids"] += ids_in_batch
        if len(self.state["processed_ids"]) > 5000:
            self.state["processed_ids"] = self.state["processed_ids"][-2500:]

        self._save_state()
        top_global = aggregated.most_common(20)
        return top_global

    # ---------- concept helpers ----------
    def _extract_concepts(self, text: str) -> List[str]:
        caps = [match.group(0) for match in CAP_WORD.finditer(text)]
        caps_norm = [cap.strip("'").lower() for cap in caps if len(cap) >= 3]

        lowered = text.lower()
        tokens = [token for token in TOKEN.findall(lowered) if token not in BASIC_STOP]

        bigrams: List[str] = []
        for idx in range(len(tokens) - 1):
            w1, w2 = tokens[idx], tokens[idx + 1]
            if w1 in BASIC_STOP or w2 in BASIC_STOP:
                continue
            bigrams.append(f"{w1} {w2}")

        concepts = tokens + bigrams + caps_norm + caps_norm
        return concepts

    def _score_select(
        self,
        tf: Counter,
        df: Dict[str, int],
        total_docs: int,
        k: int = 6,
    ) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for term, freq in tf.items():
            idf = math.log(1 + (total_docs / max(1, df.get(term, 1))))
            scored.append((term, freq * idf))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def _update_index(
        self,
        top: List[Tuple[str, float]],
        mem_id: Optional[str],
        text: str,
    ) -> None:
        now = _now()
        for concept, _ in top:
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
            if mem_id and len(entry["sources"]) < 8:
                entry["sources"].append(mem_id)
            if not entry["sample"]:
                entry["sample"] = text[:240]
            self.concept_index[concept] = entry

        with open(self.paths["concept_index"], "w", encoding="utf-8") as handle:
            json.dump(self.concept_index, handle, ensure_ascii=False, indent=2)

    def _update_graph(self, top: List[Tuple[str, float]]) -> None:
        if not top:
            return
        for concept, _ in top:
            node = self.graph["nodes"].get(concept, {"count": 0})
            node["count"] += 1
            self.graph["nodes"][concept] = node

        concepts = [concept for concept, _ in top]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                key = f"{concepts[i]}|||{concepts[j]}"
                edge = self.graph["edges"].get(key, {"w": 0.0})
                edge["w"] += 1.0
                self.graph["edges"][key] = edge

        with open(self.paths["concept_graph"], "w", encoding="utf-8") as handle:
            json.dump(self.graph, handle, ensure_ascii=False)

    def _log_event(self, mem_id: Optional[str], meta: Dict[str, Any], top: List[Tuple[str, float]]) -> None:
        _safe_jsonl_append(
            self.paths["concept_events"],
            {
                "t": _now(),
                "memory_id": mem_id,
                "top_concepts": top,
                "kind": meta.get("kind"),
                "meta": meta.get("metadata", {}),
            },
        )

    def _log_memory(self, mem_id: Optional[str], top: List[Tuple[str, float]]) -> None:
        memory = self.bound.get("memory")
        if mem_id and memory and hasattr(memory, "add_memory") and top:
            try:
                memory.add_memory(
                    kind="concept_extracted",
                    content="; ".join(concept for concept, _ in top),
                    metadata={"source_memory": mem_id, "concepts": top},
                )
            except Exception:
                pass

    def _update_store(self, mem_id: Optional[str], top: List[Tuple[str, float]]) -> None:
        if not self.store or not top:
            return
        max_score = top[0][1] if top else 1.0
        concept_ids: Dict[str, str] = {}
        for concept, score in top:
            support_delta = float(score / max_score) if max_score else 0.0
            salience_delta = 0.1 * support_delta
            obj = self.store.upsert_concept(
                label=concept,
                support_delta=support_delta,
                salience_delta=salience_delta,
                example_mem_id=mem_id,
                confidence=0.6,
            )
            concept_ids[concept] = obj.id
        concepts = [concept for concept, _ in top]
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.store.upsert_relation(
                    concept_ids.get(concepts[i]) or concepts[i],
                    concept_ids.get(concepts[j]) or concepts[j],
                    "related_to",
                    weight_delta=0.1,
                    mem_id=mem_id,
                    confidence=0.6,
                )
        self.store.save()

    # ---------- persistence helpers ----------
    def _save_state(self) -> None:
        self.state["last_run"] = _now()
        with open(self.paths["state"], "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, ensure_ascii=False, indent=2)

    # ---------- memory access ----------
    def _collect_memories(self, memory_system: Any, limit: int) -> List[Dict[str, Any]]:
        if memory_system is None:
            return self._fallback_read_files(limit)
        try:
            if hasattr(memory_system, "get_recent_memories"):
                res = memory_system.get_recent_memories(n=limit)
                if res:
                    return list(res)
        except Exception:
            pass
        try:
            if hasattr(memory_system, "iter_memories"):
                result: List[Dict[str, Any]] = []
                for item in memory_system.iter_memories():
                    result.append(item)
                    if len(result) >= limit:
                        break
                if result:
                    return result
        except Exception:
            pass
        return self._fallback_read_files(limit)

    def _fallback_read_files(self, limit: int) -> List[Dict[str, Any]]:
        root = os.path.join(self.data_dir, "memories")
        if not os.path.isdir(root):
            return []
        files = sorted(glob.glob(os.path.join(root, "*.json")), reverse=True)
        out: List[Dict[str, Any]] = []
        for path in files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    out.append(json.load(handle))
            except Exception:
                continue
        return out

    def _prepare_documents(self, memories: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Optional[str], str, Dict[str, Any]]]:
        for memory in memories:
            mem_id = memory.get("id") or memory.get("_id") or memory.get("memory_id")
            if mem_id and mem_id in self.state["processed_ids"]:
                continue
            text = _memory_to_text(memory)
            if not text:
                continue
            yield mem_id, text, memory
