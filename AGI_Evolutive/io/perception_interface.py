import os, json, time, uuid, glob
from typing import Any, Dict, List, Optional


def _now():
    return time.time()


class PerceptionInterface:
    """
    - Scrute ./inbox pour importer des fichiers comme perceptions
    - Enregistre les tours de dialogue en mémoire (ingest_user_message)
    - Écrit un journal JSONL data/perception_log.jsonl
    - Fait remonter des 'percepts' simples à Metacog/Émotions (optionnel)
    """

    def __init__(self,
                 inbox_dir: str = "inbox",
                 path_log: str = "data/perception_log.jsonl",
                 index_path: str = "data/perception_index.json"):
        os.makedirs(os.path.dirname(path_log), exist_ok=True)
        os.makedirs(inbox_dir, exist_ok=True)
        self.inbox_dir = inbox_dir
        self.path_log = path_log
        self.index_path = index_path
        self.bound = {
            "arch": None, "memory": None, "metacog": None, "emotions": None, "language": None
        }
        self._index = self._load_index()
        self.scan_interval = 3.0
        self._last_scan = 0.0

    def bind(self, arch=None, memory=None, metacog=None, emotions=None, language=None):
        self.bound.update({"arch": arch, "memory": memory, "metacog": metacog, "emotions": emotions, "language": language})

    def step(self, force: bool=False):
        now = time.time()
        if not force and (now - self._last_scan < self.scan_interval):
            return
        self._last_scan = now
        self._scan_inbox()

    # ---------- Inbox ----------
    def _scan_inbox(self):
        files = glob.glob(os.path.join(self.inbox_dir, "*"))
        new_files = [f for f in files if f not in self._index.get("seen_files", [])]
        if not new_files:
            return
        for path in new_files:
            self._ingest_file(path)
            self._index["seen_files"].append(path)
        self._save_index()

    def _ingest_file(self, path: str):
        memory = self.bound.get("memory")
        emotions = self.bound.get("emotions")

        meta = {
            "source": "inbox",
            "filename": os.path.basename(path),
            "size": os.path.getsize(path),
            "mtype": self._guess_mtype(path),
            "ingested_at": _now()
        }
        content_text = ""
        try:
            # only text-ish for simplicity
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content_text = f.read()[:100000]
        except Exception:
            content_text = f"(binaire) {meta['filename']}"

        self._log({"kind": "inbox_file", "meta": meta})

        if memory and hasattr(memory, "add_memory"):
            try:
                memory.add_memory(kind="perception_inbox",
                                  content=content_text,
                                  metadata=meta)
            except Exception:
                pass

        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(kind="perceived_input", intensity=0.2, arousal_hint=0.1)
            except Exception:
                pass

    def _guess_mtype(self, path: str) -> str:
        name = path.lower()
        if name.endswith((".txt", ".md", ".json", ".log")):
            return "text"
        if name.endswith((".png", ".jpg", ".jpeg", ".gif")):
            return "image"
        if name.endswith((".pdf",)):
            return "pdf"
        return "blob"

    # ---------- Dialogue ----------
    def ingest_user_message(self, text: str, speaker: str = "user", meta: Optional[Dict[str, Any]] = None):
        record = {"kind": "dialogue_turn", "speaker": speaker, "text": text, "t": _now(), "meta": meta or {}}
        self._log(record)

        memory = self.bound.get("memory")
        if memory and hasattr(memory, "add_memory"):
            try:
                memory.add_memory(kind="dialogue_turn", content=text, metadata={"speaker": speaker, **(meta or {})})
            except Exception:
                pass

        # petit signal émotionnel (arousal léger)
        emotions = self.bound.get("emotions")
        if emotions and hasattr(emotions, "register_emotion_event"):
            try:
                emotions.register_emotion_event(kind="dialogue_input", intensity=0.25, arousal_hint=0.15)
            except Exception:
                pass

    # ---------- utils ----------
    def _log(self, rec: Dict[str, Any]):
        rec["logged_at"] = _now()
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_path):
            return {"seen_files": []}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"seen_files": []}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
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
