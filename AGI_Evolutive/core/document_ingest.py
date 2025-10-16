
# core/document_ingest.py
"""
DocumentIngest: intègre les documents de ./inbox dans la mémoire.
- Parse des fichiers texte/markdown/json (binaire -> ignoré)
- Crée des traces mnésiques épisodiques avec associations légères
- Hook simple à appeler dans la boucle
"""
import os, json, time, glob, hashlib
from typing import Dict, Any

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

class DocumentIngest:
    def __init__(self, arch, inbox_dir: str):
        self.arch = arch
        self.inbox_dir = os.path.abspath(inbox_dir)
        os.makedirs(self.inbox_dir, exist_ok=True)
        self._index = {}  # filename -> last_hash
    
    def scan(self) -> Dict[str, Any]:
        docs = {}
        for path in glob.glob(os.path.join(self.inbox_dir, "*")):
            name = os.path.basename(path)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs[name] = content
            except Exception:
                continue
        return docs
    
    def integrate(self):
        mem = getattr(self.arch, "memory", None)
        if mem is None:
            return 0
        docs = self.scan()
        added = 0
        for name, content in docs.items():
            try:
                if hasattr(self.arch, "style_observer"):
                    self.arch.style_observer.observe_text(
                        content, source=f"inbox:{name}", channel="inbox"
                    )
            except Exception:
                pass
            h = _hash(content[:10000])
            if self._index.get(name) == h:
                continue  # déjà intégré
            # créer une trace simple en mémoire épisodique
            try:
                trace = {
                    "id": f"doc::{name}::{h[:8]}",
                    "content": content[:200000],
                    "memory_type": getattr(mem, "MemoryType", None).EPISODIC if hasattr(mem, "MemoryType") else "ep",
                    "strength": 0.6,
                    "accessibility": 0.7,
                    "valence": 0.0,
                    "timestamp": time.time(),
                    "context": {"source": "inbox", "filename": name},
                    "associations": [],
                    "consolidation_state": "LABILE",
                    "last_accessed": time.time(),
                    "access_count": 0,
                }
            except Exception:
                trace = {"id": f"doc::{name}::{h[:8]}", "content": content[:200000], "timestamp": time.time()}
            # Stocker dans long_term_memory épisodique si possible
            try:
                LTM = mem.long_term_memory
                key = trace["id"]
                mem_type = getattr(mem, "MemoryType", None).EPISODIC if hasattr(mem, "MemoryType") else None
                if mem_type is not None:
                    LTM.setdefault(mem_type, {})[key] = trace
                else:
                    # fallback: stocker sur une clé 'EPISODIC'
                    LTM.setdefault("EPISODIC", {})[key] = trace
                mem.memory_metadata["total_memories"] = mem.memory_metadata.get("total_memories", 0) + 1
            except Exception:
                pass

            # --- INDUCTION DE RÈGLES SOCIALES ---
            try:
                from AGI_Evolutive.social.interaction_miner import InteractionMiner
                from AGI_Evolutive.social.interaction_rule import InteractionRule

                self.arch.interaction_miner = getattr(self.arch, "interaction_miner", InteractionMiner(self.arch))
                rules = self.arch.interaction_miner.mine_text(content, source=f"inbox:{name}")

                # seuil bas pour enregistrer, moyen/haut pour proposer validation
                for rule in rules:
                    self.arch.memory.add_memory(rule.to_dict())

                    # (optionnel) goal de validation quand confiance moyenne
                    if 0.55 <= float(rule.confidence) < 0.75:
                        gid = f"validate_rule::{rule.id}"
                        self.arch.planner.ensure_goal(gid, f"Valider règle sociale {rule.id}", priority=0.62)
                        # pipeline simple : simuler ou chercher contre-exemples
                        self.arch.planner.add_action_step(gid, "simulate_dialogue", {"rule_id": rule.id}, priority=0.60)
                        self.arch.planner.add_action_step(gid, "search_counterexample", {"rule_id": rule.id}, priority=0.58)
            except Exception as e:
                # ne casse jamais l'ingestion si la mine échoue
                try:
                    self.arch.memory.add_memory({"kind": "warn", "text": f"interaction_miner_failed:{e}"})
                except Exception:
                    pass
            # --- fin induction ---
            self._index[name] = h
            added += 1
        return added
