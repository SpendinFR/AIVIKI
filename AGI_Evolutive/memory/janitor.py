"""Suppression douce des items expirés (sauf `pinned`). Safe si `delete_item` absent."""

from __future__ import annotations

import time



def run_once(memory_store) -> dict:
    now = time.time() if not hasattr(memory_store, "now") else memory_store.now()
    deleted = 0
    marked = 0
    # On parcourt large mais par lots, à adapter selon votre list_items
    for it in memory_store.list_items({"newer_than_ts": 0, "limit": 5000}):
        exp = it.get("expiry_ts")
        if exp and exp < now and not it.get("pinned"):
            if hasattr(memory_store, "delete_item"):
                try:
                    memory_store.delete_item(it["id"])  # hard delete
                    deleted += 1
                    continue
                except Exception:
                    pass
            # fallback: soft delete
            try:
                memory_store.update_item(it["id"], {"deleted": True})
                marked += 1
            except Exception:
                pass
    return {"deleted": deleted, "marked": marked}
