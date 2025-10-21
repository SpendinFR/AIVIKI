import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

_DEFAULTS: Dict[str, Any] = {
    "version": 1,
    "name": "agi_evolutive",
    "DATA_DIR": "data",
    "MEM_DIR": "data/memories",
    "PLANS_PATH": "data/plans.json",
    "SELF_PATH": "data/self_model.json",
    "SELF_VERSIONS_DIR": "data/self_model_versions",
    "HOMEOSTASIS_PATH": "data/homeostasis.json",
    "VECTOR_DIR": "data/vector_store",
    "LOGS_DIR": "logs",
    "GOALS_DAG_PATH": "logs/goals_dag.json",
    "PRIMARY_USER_NAME": "William",
    "PRIMARY_USER_ROLE": "creator",
    "MEMORY_SHARING_TRUSTED_NAMES": ["William"],
    "MEMORY_SHARING_ROLES_BY_NAME": {},
    "MEMORY_SHARING_TRUSTED_ROLES": ["creator", "owner"],
    "MEMORY_SHARING_MAX_ITEMS": 5,
}

_DIR_KEYS = ("DATA_DIR", "MEM_DIR", "SELF_VERSIONS_DIR", "VECTOR_DIR", "LOGS_DIR")

_cfg: Optional[Dict[str, Any]] = None


@lru_cache(maxsize=1)
def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from *path*, applying defaults and creating directories."""
    global _cfg

    if os.path.isabs(path):
        candidate_paths = [path]
    else:
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate_paths = [os.path.join(here, path), path]

    cfg = dict(_DEFAULTS)
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    cfg.update(json.load(fh))
                break
            except Exception:
                continue

    for key in _DIR_KEYS:
        os.makedirs(cfg[key], exist_ok=True)

    _cfg = cfg
    return _cfg


def cfg() -> Dict[str, Any]:
    """Return the cached configuration, loading it if required."""
    global _cfg
    return _cfg or load_config()
