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
import os
import json
from typing import Any, Dict, Optional

_DEFAULTS: Dict[str, Any] = {
    "DATA_DIR": "data",
    "MEM_DIR": "data/memories",
    "PLANS_PATH": "data/plans.json",
    "SELF_PATH": "data/self_model.json",
    "SELF_VERSIONS_DIR": "data/self_model_versions",
    "HOMEOSTASIS_PATH": "data/homeostasis.json",
    "VECTOR_DIR": "data/vector_store",
    "LOGS_DIR": "logs",
    "GOALS_DAG_PATH": "logs/goals_dag.json",
}

_cfg: Optional[Dict[str, Any]] = None


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from *path* and ensure core directories exist."""
    global _cfg
    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            # Silently ignore malformed configuration, keep defaults
            pass

    for key in ("DATA_DIR", "MEM_DIR", "SELF_VERSIONS_DIR", "VECTOR_DIR", "LOGS_DIR"):
        os.makedirs(cfg[key], exist_ok=True)

    _cfg = cfg
    return _cfg


def cfg() -> Dict[str, Any]:
    """Return the cached configuration, loading it if required."""
    global _cfg
    return _cfg or load_config()
