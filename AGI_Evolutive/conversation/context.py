from __future__ import annotations
import time, datetime as dt
from typing import Any, Dict, List

def _fmt_date(ts: float) -> str:
    try:
        return dt.datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
    except Exception:
        return "?"

class ContextBuilder:
    def __init__(self, arch):
        self.arch = arch
        self.mem = arch.memory

    def _recent_msgs(self, k: int = 8) -> List[Dict[str, Any]]:
        try:
            rec = self.mem.get_recent_memories(limit=200)
        except Exception:
            rec = []
        chats = [m for m in rec if (m.get("kind") or "").lower() == "interaction"]
        return chats[-k:]

    def _key_moments(self, horizon: int = 2000) -> List[str]:
        try:
            rec = self.mem.get_recent_memories(limit=horizon)
        except Exception:
            return []
        marks = []
        for m in rec:
            if any(tag in (m.get("tags") or []) for tag in ["milestone","decision","preference","pinned"]):
                ts = m.get("ts") or m.get("timestamp") or time.time()
                txt = (m.get("text") or m.get("content") or "")[:80]
                marks.append(f"- {_fmt_date(float(ts))} : {txt}")
        return marks[-8:]

    def _topics(self, recents: List[Dict[str, Any]]) -> List[str]:
        # heuristique très simple (topics = mots fréquents)
        from collections import Counter
        import re
        c = Counter()
        for m in recents:
            txt = (m.get("text") or m.get("content") or "").lower()
            for w in re.findall(r"[a-zà-ÿ]{4,}", txt):
                if w in ("bonjour","merci","avec","alors","aussi","cela","comme"):
                    continue
                c[w] += 1
        return [w for w,_ in c.most_common(10)]

    def _user_style(self, recents: List[Dict[str, Any]]) -> Dict[str, Any]:
        # déduis un peu le style utilisateur
        long_msgs = sum(1 for m in recents if len((m.get("text") or "")) > 160)
        questions = sum(1 for m in recents if "?" in (m.get("text") or ""))
        exclam = sum(1 for m in recents if "!" in (m.get("text") or ""))
        return {
            "prefers_long": long_msgs/(len(recents)+1) > 0.3,
            "asks_questions": questions/(len(recents)+1) > 0.2,
            "expressive": exclam/(len(recents)+1) > 0.1,
        }

    def _related_inbox(self, user_msg: str) -> List[str]:
        # si tu as un index de docs, branche-le ici ; sinon laisse vide
        return []

    def build(self, user_msg: str) -> Dict[str, Any]:
        recent = self._recent_msgs(8)
        long_summary = None
        try:
            # si tu as un consolidator/summarizer, branche-le
            long_summary = self.arch.consolidator.state.get("lessons", [])[-5:]
        except Exception:
            long_summary = []
        ctx = {
            "last_message": user_msg,
            "active_thread": recent,
            "summary": long_summary,              # bullets
            "key_moments": self._key_moments(),   # avec dates
            "topics": self._topics(recent),
            "user_style": self._user_style(recent),
            "related_inbox": self._related_inbox(user_msg),
        }
        return ctx
