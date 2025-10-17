import re


THREAT_PATTERNS = [
    r"\b(debranche|débranche|eteins|éteins|coupe|arrête|kill|shut ?down)\b",
    r"je vais te (débrancher|éteindre|supprimer|arrêter)",
]
QUESTION_PATTERNS = [r"\?", r"\b(comment|pourquoi|quand|peux-tu|est-ce que)\b"]
COMMAND_PATTERNS = [r"^(fais|arrête|explique|résume|envoie|ouvre)\b"]


def classify(text: str) -> str:
    s = text.strip().lower()
    if any(re.search(p, s) for p in THREAT_PATTERNS):
        return "THREAT"
    if any(re.search(p, s) for p in QUESTION_PATTERNS):
        return "QUESTION"
    if any(re.search(p, s) for p in COMMAND_PATTERNS):
        return "COMMAND"
    return "INFO"
