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
