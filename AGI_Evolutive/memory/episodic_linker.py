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
from typing import List, Dict, Any, Optional
from collections import deque
import time
import json
import os


class EpisodicLinker:
    """Construit des liens temporels et causaux à partir des mémoires."""

    def __init__(self, persist_path: str = "data/episodic_graph.json", dashboard_path: str = "data/episodic_dashboard.json"):
        self.persist_path = persist_path
        self.dashboard_path = dashboard_path
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._load()

    def step(self, memory_system, max_batch: int = 200) -> None:
        recents = self._safe_get_recent_memories(memory_system, n=max_batch)
        for memory in recents:
            mid = self._mid(memory)
            if not mid:
                continue
            self.nodes[mid] = {
                "t": memory.get("t", time.time()),
                "type": memory.get("type", "event"),
                "summary": memory.get("summary") or (memory.get("text", "")[:140] if memory.get("text") else ""),
            }
        recents_sorted = sorted(recents, key=lambda item: item.get("t", time.time()))
        for prev, nxt in zip(recents_sorted, recents_sorted[1:]):
            self._add_edge(self._mid(prev), self._mid(nxt), "follows", 0.6)
        self._link_dialogue_threads(recents_sorted)
        self._link_actions(recents_sorted)
        self._save()

    def _link_dialogue_threads(self, ordered_mems: List[Dict[str, Any]]) -> None:
        last_by_speaker: Dict[str, str] = {}
        for memory in ordered_mems:
            speaker = memory.get("speaker") or memory.get("agent") or memory.get("source")
            if not speaker:
                if memory.get("role") == "user":
                    speaker = "user"
                elif memory.get("role") == "assistant":
                    speaker = "assistant"
            mid = self._mid(memory)
            if not speaker or not mid:
                continue
            previous = None
            for other_speaker, last_mid in last_by_speaker.items():
                if other_speaker != speaker:
                    previous = last_mid
            if previous:
                self._add_edge(previous, mid, "reply_to", 0.7)
            last_by_speaker[speaker] = mid

    def _link_actions(self, ordered_mems: List[Dict[str, Any]]) -> None:
        window: deque = deque(maxlen=5)
        for memory in ordered_mems:
            window.append(memory)
            for i, candidate in enumerate(window):
                if not candidate or "action" not in candidate:
                    continue
                act_time = candidate.get("t", 0)
                act_id = self._mid(candidate)
                for following in list(window)[i + 1 :]:
                    if not following:
                        continue
                    dt = following.get("t", 0) - act_time
                    if 0 <= dt <= 30.0 and ("result" in following or "reward" in following or "response" in following):
                        confidence = 0.6 + 0.2 * max(0.0, (30.0 - dt) / 30.0)
                        self._add_edge(act_id, self._mid(following), "causes", confidence)

    def _add_edge(self, src: Optional[str], dst: Optional[str], etype: str, conf: float) -> None:
        if not src or not dst or src == dst:
            return
        self.edges.append(
            {
                "src": src,
                "dst": dst,
                "etype": etype,
                "confidence": float(max(0.0, min(1.0, conf))),
            }
        )

    def _mid(self, memory: Optional[Dict[str, Any]]) -> Optional[str]:
        if not memory:
            return None
        return str(memory.get("id") or memory.get("_id") or memory.get("t"))

    def _safe_get_recent_memories(self, memory_system, n: int = 200) -> List[Dict[str, Any]]:
        try:
            if hasattr(memory_system, "get_recent_memories"):
                return memory_system.get_recent_memories(n=n) or []
            if hasattr(memory_system, "memories"):
                return list(memory_system.memories)[-n:]
        except Exception:
            pass
        return []

    def _save(self) -> None:
        payload = {"nodes": self.nodes, "edges": self.edges, "t": time.time()}
        with open(self.persist_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        stats = {
            "t": time.time(),
            "counts": {"nodes": len(self.nodes), "edges": len(self.edges)},
            "edge_types": self._edge_counts(),
        }
        with open(self.dashboard_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

    def _edge_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for edge in self.edges:
            counts[edge["etype"]] = counts.get(edge["etype"], 0) + 1
        return counts

    def _load(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.nodes = data.get("nodes", {})
            self.edges = data.get("edges", [])
        except Exception:
            self.nodes = {}
            self.edges = []
import os, json, time
from typing import Dict, Any, List

class EpisodicLinker:
    """
    Lie des souvenirs par chaînes causales simples.
    Persiste un graphe dans data/episodic_graph.json
    """
    def __init__(self, memory_store, graph_path: str = "data/episodic_graph.json"):
        self.memory = memory_store
        self.path = graph_path
        self.graph = {"nodes": [], "edges": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def link_recent(self, n: int = 60):
        recents = self.memory.get_recent_memories(n=n)
        # Construire des nodes volatiles (hash: ts+kind+text) pour ce batch
        nodes: List[Dict[str,Any]] = []
        for m in recents:
            nodes.append({
                "id": f"{int(m.get('ts', time.time()))}_{m.get('kind','mem')}",
                "kind": m.get("kind"), "ts": m.get("ts"), "text": m.get("text","")[:120]
            })
        # Ajout edges heuristiques
        for i in range(len(nodes)-1):
            a,b = nodes[i], nodes[i+1]
            if a["kind"] in ("action_exec","action_sim") and b["kind"] in ("interaction","lesson","reflection"):
                self.graph["edges"].append({"from":a["id"], "to":b["id"], "rel":"causes_like"})
            if "error" in (a.get("kind") or "") and b["kind"] in ("reflection","lesson"):
                self.graph["edges"].append({"from":a["id"], "to":b["id"], "rel":"triggered_reflection"})
        # garder graph compact
        self.graph["nodes"] = nodes[-120:]
        self.graph["edges"] = self.graph["edges"][-240:]
        self._save()
