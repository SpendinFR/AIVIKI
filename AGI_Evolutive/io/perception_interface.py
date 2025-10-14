import os, time
from typing import Dict, Any, List

class PerceptionInterface:
    """
    Ingestion d'inputs (messages, fichiers inbox/, signaux système).
    """
    def __init__(self, memory_store, inbox_dir: str = "inbox"):
        self.memory = memory_store
        self.inbox_dir = inbox_dir
        os.makedirs(self.inbox_dir, exist_ok=True)
        self._seen_files: set = set(os.listdir(self.inbox_dir))

    def ingest_user_utterance(self, text: str, author: str = "user"):
        self.memory.add_memory({"kind":"interaction","author":author,"text":text,"ts":time.time()})

    def scan_inbox(self) -> List[str]:
        added = []
        for fn in os.listdir(self.inbox_dir):
            if fn in self._seen_files: continue
            path = os.path.join(self.inbox_dir, fn)
            try:
                size = os.path.getsize(path)
                self.memory.add_memory({
                    "kind":"document_ingested","text":f"Fichier {fn} ({size} bytes)","path":path,"ts":time.time()
                })
                self._seen_files.add(fn)
                added.append(fn)
            except Exception:
                pass
        return added
