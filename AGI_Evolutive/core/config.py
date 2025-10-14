import json
import os
from functools import lru_cache

_DEFAULT_CONFIG = {
    "version": 1,
    "name": "agi_evolutive"
}

@lru_cache(maxsize=1)
def load_config(path: str = "config.json") -> dict:
    """Loads a simple JSON configuration file, returning defaults if absent."""
    if os.path.isabs(path):
        candidate_paths = [path]
    else:
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate_paths = [os.path.join(here, path), path]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    return data
            except Exception:
                break
    return dict(_DEFAULT_CONFIG)
