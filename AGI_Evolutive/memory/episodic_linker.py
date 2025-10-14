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
