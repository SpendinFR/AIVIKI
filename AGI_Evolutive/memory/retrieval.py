import time
from typing import Dict, Any, Optional, List, Tuple

try:
    from config.memory_flags import ENABLE_RERANKING  # type: ignore
except Exception:
    ENABLE_RERANKING = True


try:  # pragma: no cover - optional integration
    from memory.salience_scorer import SalienceScorer  # type: ignore
    from memory.preferences_adapter import PreferencesAdapter  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    SalienceScorer = None
    PreferencesAdapter = None


from .encoders import TinyEncoder
from .indexing import InMemoryIndex
from .vector_store import VectorStore


def _rerank_candidates(
    candidates: List[Dict[str, Any]],
    *,
    salience_scorer: Optional["SalienceScorer"] = None,
    preferences: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Enrichit et re-score les candidats.

    Cette étape reste optionnelle : si le scorer ou les préférences ne sont pas
    disponibles, les éléments sont renvoyés tels quels.
    """

    if not candidates:
        return candidates
    if salience_scorer is None:
        return candidates

    for record in candidates:
        salience = record.get("salience")
        if salience is None:
            try:
                salience = float(salience_scorer.score(record))  # type: ignore[arg-type]
            except Exception:
                salience = 0.0
            record["salience"] = salience

        affinity = 0.0
        if preferences is not None:
            try:
                concepts = record.get("concepts", [])
                tags = record.get("tags", [])
                affinity = float(preferences.get_affinity(concepts, tags))
            except Exception:
                affinity = 0.0
        record["_affinity"] = affinity

        recency = 0.0
        try:
            now = time.time()
            ts = float(record.get("ts", now))
            recency = max(0.0, min(1.0, 1.0 - ((now - ts) / (30 * 24 * 3600))))
        except Exception:
            pass
        record["_recency"] = recency

        lexical_score = float(record.get("lexical", 0.0))
        vector_score = float(record.get("vector", 0.0))
        record["final"] = (
            0.45 * lexical_score
            + 0.35 * vector_score
            + 0.10 * float(record.get("salience", 0.0))
            + 0.07 * recency
            + 0.03 * affinity
        )

    return candidates


class MemoryRetrieval:
    """
    Façade Retrieval:
    - add_interaction(user, agent, extra)
    - add_document(text, title, source)
    - search_text(query, top_k)
    """

    def __init__(
        self,
        encoder: Optional[TinyEncoder] = None,
        index: Optional[InMemoryIndex] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        salience_scorer: Optional["SalienceScorer"] = None,
        preferences: Optional[Any] = None,
    ):
        self.encoder = encoder or TinyEncoder()
        self.index = index or InMemoryIndex(self.encoder)
        self.vector_store = vector_store
        self.salience_scorer = salience_scorer
        self.preferences = None

        if preferences is not None:
            self.preferences = preferences
            if PreferencesAdapter and not hasattr(preferences, "get_affinity"):
                try:
                    self.preferences = PreferencesAdapter(preferences)  # type: ignore[call-arg]
                except Exception:
                    self.preferences = preferences

    # -------- Ajout ----------
    def add_interaction(self, user: str, agent: str, extra: Optional[Dict[str, Any]] = None) -> int:
        text = f"[USER] {user}\n[AGENT] {agent}"
        meta = {"type": "interaction", "ts": time.time()}
        if extra:
            meta.update(extra)
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"interaction::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    def add_document(self, text: str, title: Optional[str] = None, source: Optional[str] = None) -> int:
        meta = {"type": "doc"}
        if title:
            meta["title"] = title
        if source:
            meta["source"] = source
        meta["ts"] = time.time()
        doc_id = self.index.add_document(text, meta=meta)
        if self.vector_store:
            try:
                self.vector_store.upsert(f"doc::{doc_id}", text)
            except Exception:
                pass
        return doc_id

    # -------- Recherche ----------
    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        lexical_hits: List[Dict[str, Any]] = []
        try:
            lexical_hits = self.index.search_text(query, top_k=top_k)
        except Exception:
            lexical_hits = []

        combined: Dict[int, Dict[str, Any]] = {}

        for entry in lexical_hits:
            doc_id = entry.get("id")
            if doc_id is None:
                continue
            doc_key = int(doc_id)
            payload = combined.setdefault(
                doc_key,
                {
                    "id": doc_key,
                    "text": entry.get("text", ""),
                    "meta": dict(entry.get("meta", {})),
                },
            )
            payload["lexical"] = float(entry.get("score", 0.0))
            payload.setdefault("ts", payload.get("meta", {}).get("ts"))

        vector_hits: List[Tuple[str, float]] = []
        if self.vector_store:
            try:
                vector_hits = self.vector_store.search(query or "", k=top_k)
            except Exception:
                vector_hits = []

        for doc_identifier, score in vector_hits:
            kind, doc_record = self._resolve_hit(doc_identifier)
            if not doc_record:
                continue
            doc_key = int(doc_record["id"])
            payload = combined.setdefault(
                doc_key,
                {
                    "id": doc_key,
                    "text": doc_record.get("text", ""),
                    "meta": dict(doc_record.get("meta", {})),
                },
            )
            payload["vector"] = max(float(score), float(payload.get("vector", 0.0)))
            meta = dict(doc_record.get("meta", {}))
            if meta:
                payload.setdefault("meta", {}).update({k: v for k, v in meta.items() if k not in payload["meta"]})
            if kind:
                payload.setdefault("meta", {})["vector_kind"] = kind
            payload.setdefault("ts", payload.get("meta", {}).get("ts"))

        candidates = list(combined.values())

        if ENABLE_RERANKING and self.salience_scorer:
            candidates = _rerank_candidates(
                candidates,
                salience_scorer=self.salience_scorer,
                preferences=self.preferences,
            )

        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            final_score = candidate.get("final")
            if final_score is None:
                final_score = max(
                    float(candidate.get("lexical", 0.0)),
                    float(candidate.get("vector", 0.0)),
                )
                candidate["final"] = final_score
            candidate["score"] = float(round(candidate.get("final", final_score), 4))
            results.append(candidate)

        results.sort(key=lambda item: item.get("final", item.get("score", 0.0)), reverse=True)

        if not results:
            return lexical_hits

        return results[: max(1, top_k)]

    # -------- Persistance (optionnelle) ----------
    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)

    # -------- Helpers ----------
    def _resolve_hit(self, identifier: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not identifier:
            return None, None
        raw = identifier
        kind: Optional[str] = None
        if "::" in raw:
            kind, raw = raw.split("::", 1)
        try:
            doc_id = int(raw)
        except ValueError:
            return kind, None
        record = self.index.get_document(doc_id)
        return kind, record
