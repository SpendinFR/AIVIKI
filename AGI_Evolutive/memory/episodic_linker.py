import os
import json
import time
import glob
from typing import Any, Dict, List, Optional


def _now():
    return time.time()


CAUSE_HINTS = [
    "parce que",
    "car",
    "donc",
    "alors",
    "ainsi",
    "du coup",
    "because",
    "therefore",
    "so that",
    "so ",
    "hence",
]
REL_NEXT = "NEXT"
REL_CAUSES = "CAUSES"
REL_REFERS = "REFERS_TO"
REL_SUPPORTS = "SUPPORTS"
REL_CONTRADICTS = "CONTRADICTS"


class EpisodicLinker:
    """
    - Regroupe des mémoires proches dans le temps en 'épisodes' (fenêtre)
    - Ajoute des liens temporels/causaux simples entre mémoires
    - Écrit episodes.jsonl + backlinks (memory_backlinks.json)
    - Pousse des mémoires 'episode_summary' si possible
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.paths = {
            "episodes": os.path.join(self.data_dir, "episodes.jsonl"),
            "backlinks": os.path.join(self.data_dir, "memory_backlinks.json"),
            "state": os.path.join(self.data_dir, "episodic_state.json"),
        }
        self.bound = {"memory": None, "language": None, "metacog": None, "emotions": None}

        self.state = self._load(
            self.paths["state"],
            {"last_run": 0.0, "processed_ids": [], "last_episode_id": 0},
        )
        self.backlinks = self._load(self.paths["backlinks"], {})  # mem_id -> [{to, rel}]

        self.period_s = 20.0
        self._last_step = 0.0
        self.window_s = 15 * 60  # 15 minutes pour grouper en épisode

    def bind(self, memory=None, language=None, metacog=None, emotions=None):
        self.bound.update(
            {
                "memory": memory,
                "language": language,
                "metacog": metacog,
                "emotions": emotions,
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

        # trier par timestamp si dispo
        def _ts(memory):
            metadata = memory.get("metadata", {})
            return float(metadata.get("timestamp", memory.get("t", memory.get("ts", _now()))))

        mems = sorted(mems, key=_ts)

        # filtrer déjà traités
        batch: List[Dict[str, Any]] = []
        for memory in mems:
            mid = memory.get("id") or memory.get("_id") or memory.get("memory_id")
            if not mid:
                continue
            if mid in self.state["processed_ids"]:
                continue
            batch.append(memory)

        if not batch:
            return

        # grouper en épisodes
        episodes = self._group_into_episodes(batch, key_ts=_ts)
        for episode in episodes:
            ep_id = self._next_episode_id()
            rels = self._link_relations(episode["memories"])
            summary = self._summarize_episode(episode["memories"])
            record = {
                "episode_id": ep_id,
                "start": episode["start"],
                "end": episode["end"],
                "size": len(episode["memories"]),
                "memory_ids": [
                    memory.get("id") or memory.get("_id") or memory.get("memory_id")
                    for memory in episode["memories"]
                ],
                "relations": rels,
                "summary": summary,
            }
            self._append_jsonl(self.paths["episodes"], record)
            self._apply_backlinks(record["relations"])
            self._emit_episode_memory(summary, record)

        # marquer traités
        new_ids = [
            memory.get("id") or memory.get("_id") or memory.get("memory_id")
            for memory in batch
        ]
        self.state["processed_ids"] += new_ids
        if len(self.state["processed_ids"]) > 5000:
            self.state["processed_ids"] = self.state["processed_ids"][-2500:]
        self.state["last_run"] = _now()
        self._save(self.paths["state"], self.state)
        self._save(self.paths["backlinks"], self.backlinks)

    # ---------- grouping ----------
    def _group_into_episodes(self, mems: List[Dict[str, Any]], key_ts) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        cur: List[Dict[str, Any]] = []
        cur_start: Optional[float] = None
        last_t: Optional[float] = None

        for memory in mems:
            timestamp = key_ts(memory)
            if last_t is None:
                cur = [memory]
                cur_start = timestamp
                last_t = timestamp
                continue
            if timestamp - last_t <= self.window_s:
                cur.append(memory)
                last_t = timestamp
            else:
                episodes.append({"start": cur_start, "end": last_t, "memories": cur[:]})
                cur = [memory]
                cur_start = timestamp
                last_t = timestamp
        if cur:
            episodes.append({"start": cur_start, "end": last_t, "memories": cur[:]})
        return episodes

    # ---------- linking ----------
    def _link_relations(self, mems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []
        ids = [memory.get("id") or memory.get("_id") or memory.get("memory_id") for memory in mems]
        texts = [str(memory.get("content", "")) for memory in mems]

        # NEXT relations (séquentiel)
        for i in range(len(ids) - 1):
            rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_NEXT})

        # heuristique CAUSES/REFERS via texte
        for i in range(len(ids)):
            t_i = texts[i].lower()
            for j in range(i + 1, min(i + 5, len(ids))):
                t_j = texts[j].lower()
                if any(hint in t_j for hint in CAUSE_HINTS) or "car " in t_j or "parce " in t_j:
                    rels.append({"src": ids[i], "dst": ids[j], "rel": REL_CAUSES})
                if "voir" in t_j or "cf." in t_j or "réf" in t_j or "ref " in t_j:
                    rels.append({"src": ids[i], "dst": ids[j], "rel": REL_REFERS})

        # petites contradictions/soutiens naïfs
        for i in range(len(ids) - 1):
            a = texts[i].lower()
            b = texts[i + 1].lower()
            if any(word in a for word in ["oui", "vrai", "possible"]) and any(
                word in b for word in ["non", "faux", "impossible"]
            ):
                rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_CONTRADICTS})
            if ("appris" in b or "confirm" in b or "conclu" in b) and len(a) > 10:
                rels.append({"src": ids[i], "dst": ids[i + 1], "rel": REL_SUPPORTS})

        return rels

    # ---------- summary ----------
    def _summarize_episode(self, mems: List[Dict[str, Any]]) -> str:
        lang = self.bound.get("language")
        # tenter un résumé via module language
        if lang and hasattr(lang, "summarize"):
            try:
                return lang.summarize([memory.get("content", "") for memory in mems])
            except Exception:
                pass
        # fallback minimal : 1ère et dernière phrases tronquées
        first = (mems[0].get("content") or "")[:180]
        last = (mems[-1].get("content") or "")[:180]
        return f"Épisode ({len(mems)} mémoires) — début: {first} ... fin: {last}"

    # ---------- apply backlinks + emit episode mem ----------
    def _apply_backlinks(self, relations: List[Dict[str, Any]]):
        mem = self.bound.get("memory")
        for relation in relations:
            src, dst, rel = relation["src"], relation["dst"], relation["rel"]
            if not src or not dst:
                continue
            self.backlinks.setdefault(dst, [])
            self.backlinks[dst].append({"from": src, "rel": rel, "t": _now()})
            # enregistre une mémoire "lien" optionnelle
            if mem and hasattr(mem, "add_memory"):
                try:
                    mem.add_memory(
                        kind="memory_link",
                        content=f"{src} -> {dst} [{rel}]",
                        metadata={"from": src, "to": dst, "relation": rel},
                    )
                except Exception:
                    pass

    def _emit_episode_memory(self, summary: str, record: Dict[str, Any]):
        mem = self.bound.get("memory")
        if mem and hasattr(mem, "add_memory"):
            try:
                mem.add_memory(
                    kind="episode_summary",
                    content=summary,
                    metadata={
                        "episode_id": record["episode_id"],
                        "start": record["start"],
                        "end": record["end"],
                        "size": record["size"],
                        "memory_ids": record["memory_ids"],
                    },
                )
            except Exception:
                pass

    # ---------- io ----------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load(self, path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save(self, path: str, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ---------- memory fetch ----------
    def _fetch_recent_memories(self, limit: int = 400) -> List[Dict[str, Any]]:
        mem = self.bound.get("memory")
        if mem is None:
            return self._fallback_read_files(limit)
        try:
            if hasattr(mem, "get_recent_memories"):
                res = mem.get_recent_memories(n=limit) or []
                return res
        except Exception:
            pass
        try:
            if hasattr(mem, "iter_memories"):
                res = []
                for memory in mem.iter_memories():
                    res.append(memory)
                    if len(res) >= limit:
                        break
                return res
        except Exception:
            pass
        return self._fallback_read_files(limit)

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

    # ---------- helpers ----------
    def _next_episode_id(self) -> int:
        self.state["last_episode_id"] += 1
        return self.state["last_episode_id"]
