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
