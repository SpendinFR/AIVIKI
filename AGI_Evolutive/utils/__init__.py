import datetime
import json
import os
from typing import Any

from AGI_Evolutive.utils.jsonsafe import json_sanitize

__all__ = ["now_iso", "safe_write_json", "json_sanitize"]


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_sanitize(obj), handle, ensure_ascii=False, indent=2)
