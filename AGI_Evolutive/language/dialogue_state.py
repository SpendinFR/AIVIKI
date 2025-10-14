from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class DialogueState:
    conversation_id: str = "default"
    turn_index: int = 0
    last_speaker: str = "user"
    user_profile: Dict[str, Any] = field(default_factory=dict)
    known_entities: Dict[str, Any] = field(default_factory=dict)
    pending_questions: List[str] = field(default_factory=list)
    recent_frames: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def update_with_frame(self, frame: Dict[str, Any]):
        self.turn_index += 1
        self.last_speaker = "user"
        self.recent_frames.append(frame)
        if len(self.recent_frames) > 20:
            self.recent_frames = self.recent_frames[-20:]

        # Retenir éventuels nouveaux référents
        for k, v in frame.get("slots", {}).items():
            if isinstance(v, str) and len(v) <= 128:
                self.known_entities[k] = v

    def add_pending_question(self, q: str):
        if q and q not in self.pending_questions:
            self.pending_questions.append(q)
            if len(self.pending_questions) > 5:
                self.pending_questions = self.pending_questions[-5:]

    def consume_pending_questions(self, max_q: int = 2) -> List[str]:
        qs = self.pending_questions[:max_q]
        self.pending_questions = self.pending_questions[max_q:]
        return qs

    def remember_unknown_term(self, term: str):
        if "unknown_terms" not in self.user_profile:
            self.user_profile["unknown_terms"] = []
        if term not in self.user_profile["unknown_terms"]:
            self.user_profile["unknown_terms"].append(term)
            if len(self.user_profile["unknown_terms"]) > 50:
                self.user_profile["unknown_terms"] = self.user_profile["unknown_terms"][-50:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "last_speaker": self.last_speaker,
            "user_profile": self.user_profile,
            "known_entities": self.known_entities,
            "pending_questions": list(self.pending_questions),
            "recent_frames": list(self.recent_frames),
            "created_at": self.created_at,
        }
