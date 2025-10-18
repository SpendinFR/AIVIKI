from __future__ import annotations

import re
from typing import Dict, List


class StyleCritic:
    """Critique légère du style pour post-traiter les réponses."""

    def __init__(self, max_chars: int = 1200):
        self.max_chars = int(max_chars)

    def analyze(self, text: str) -> Dict[str, object]:
        sample = (text or "").strip()
        length = len(sample)
        lines: List[str] = sample.splitlines() or [""]
        flags: List[str] = []

        if length > self.max_chars:
            flags.append("trim")
        if any(len(line) > 320 for line in lines):
            flags.append("long_line")
        if "  " in sample:
            flags.append("double_space")

        return {
            "length": length,
            "line_count": len(lines),
            "flags": flags,
        }

    def rewrite(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip()
            if not cleaned.endswith("."):
                cleaned = cleaned.rstrip("…") + "…"
        return cleaned
