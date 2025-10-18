import os
import hashlib
from typing import Iterable

from . import DATA_DIR, _json_load, _json_save

CACHE = os.path.join(DATA_DIR, "inbox_cache.json")


def _hash_path(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8", errors="ignore")).hexdigest()


def ingest_inbox_paths(paths: Iterable[str], *, arch) -> int:
    """
    Ingère des fichiers texte depuis l'inbox :
      - alimente le Lexicon
      - observe le style
      - ajoute des citations à la QuoteMemory

    Un cache persistant empêche la ré-ingestion inutile.
    """

    cache = _json_load(CACHE, {})
    added = 0
    qm = getattr(arch, "quote_memory", None) or getattr(
        getattr(arch, "voice_profile", None), "quote_memory", None
    )
    lex = getattr(arch, "lexicon", None)
    style_obs = getattr(arch, "style_observer", None)

    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        h = _hash_path(p)
        if cache.get(h):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = (line or "").strip()
                    if not text:
                        continue
                    if lex:
                        try:
                            lex.add_from_text(text, liked=False)
                        except Exception:
                            pass
                    if style_obs:
                        try:
                            style_obs.observe(text)
                        except Exception:
                            pass
            if qm:
                try:
                    qm.ingest_file_units(p, liked=True)
                except Exception:
                    pass
            cache[h] = {"path": p}
            added += 1
        except Exception:
            continue

    _json_save(CACHE, cache)
    if qm:
        try:
            qm.save()
        except Exception:
            pass
    return added
