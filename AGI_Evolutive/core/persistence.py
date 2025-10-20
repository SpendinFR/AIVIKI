
# core/persistence.py
"""
PersistenceManager: sauvegarde/chargement robuste de l'état de l'AGI en local.
- Sauvegarde automatique à intervalle régulier et sur demande
- Reprise à chaud: si un snapshot existe, on recharge lors de l'initialisation
- Sérialisation prudente: on filtre les objets non-sérialisables
"""
import hashlib
import json
import os
import pickle
import time
import types
import inspect
from datetime import datetime
from typing import Any, Dict, Optional

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".agi_state")
DEFAULT_DIR = os.path.abspath(DEFAULT_DIR)
DEFAULT_FILE = os.path.join(DEFAULT_DIR, "snapshot.pkl")

def _is_picklable(x):
    try:
        pickle.dumps(x)
        return True
    except Exception:
        return False

def _to_state(obj):
    """
    Essaie d'extraire un dict sérialisable depuis 'obj'.
    - Si l'objet expose .to_state(), on l'utilise.
    - Sinon, on tente __dict__ en filtrant les valeurs non picklables.
    """
    if hasattr(obj, "to_state") and callable(getattr(obj, "to_state")):
        try:
            state = obj.to_state()
            if _is_picklable(state):
                return state
        except Exception:
            pass
    d = {}
    # fallback sur __dict__ si dispo
    src = getattr(obj, "__dict__", {})
    for k, v in src.items():
        # ignorer méthodes, modules, fonctions, générateurs et coroutines
        if isinstance(v, (types.ModuleType, types.FunctionType, types.GeneratorType)):
            continue
        if inspect.isroutine(v) or inspect.isclass(v):
            continue
        if _is_picklable(v):
            d[k] = v
        else:
            d[k] = f"<non_picklable:{type(v).__name__}>"
    return d

def _from_state(obj, state: Dict[str, Any]):
    """
    Restaure un état simple dans l'objet (best-effort).
    - Si l'objet expose .from_state(state), on l'utilise.
    - Sinon on met à jour __dict__ avec les clés existantes uniquement.
    """
    if hasattr(obj, "from_state") and callable(getattr(obj, "from_state")):
        try:
            obj.from_state(state)
            return
        except Exception:
            pass
    if not hasattr(obj, "__dict__"):
        return
    for k, v in state.items():
        try:
            setattr(obj, k, v)
        except Exception:
            pass

class PersistenceManager:
    def __init__(self, arch, directory: str = DEFAULT_DIR, filename: str = DEFAULT_FILE):
        self.arch = arch
        self.directory = os.path.abspath(directory)
        self.filename = os.path.abspath(filename)
        self.autosave_interval = 60.0  # secondes
        self._last_save = time.time()
        os.makedirs(self.directory, exist_ok=True)
        self.history_dir = os.path.join(self.directory, "history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_retention = 20
        self._last_snapshot_meta: Optional[Dict[str, Any]] = None
        self._last_snapshot_hash: Optional[str] = None
        self._last_drift: Optional[Dict[str, Any]] = None

    def make_snapshot(self) -> Dict[str, Any]:
        subs = [
            "memory","perception","reasoning","goals","emotions",
            "learning","metacognition","creativity","world_model","language"
        ]
        snap = {"timestamp": time.time(), "version": 1}
        for name in subs:
            comp = getattr(self.arch, name, None)
            if comp is None:
                snap[name] = None
            else:
                snap[name] = _to_state(comp)
        return snap
    
    def save(self):
        snap = self.make_snapshot()
        payload = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
        tmpfile = self.filename + ".tmp"
        with open(tmpfile, "wb") as f:
            f.write(payload)
        os.replace(tmpfile, self.filename)
        self._last_save = time.time()
        digest = hashlib.sha256(payload).hexdigest()
        summary = self._summarize_snapshot(snap, digest)
        prev_summary = self._last_snapshot_meta
        self._last_drift = self._compute_drift(prev_summary, summary)
        self._last_snapshot_meta = summary
        previous_hash = self._last_snapshot_hash
        self._last_snapshot_hash = digest
        if digest != previous_hash:
            self._record_history(payload, summary)
            self._prune_history()
        return self.filename

    def load(self) -> bool:
        if not os.path.exists(self.filename):
            return False
        try:
            with open(self.filename, "rb") as f:
                snap = pickle.load(f)
            payload = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
            for name, state in snap.items():
                if name in ("timestamp", "version"):
                    continue
                comp = getattr(self.arch, name, None)
                if comp is not None and isinstance(state, dict):
                    _from_state(comp, state)
            digest = hashlib.sha256(payload).hexdigest()
            self._last_snapshot_meta = self._summarize_snapshot(snap, digest)
            self._last_snapshot_hash = digest
            self._last_drift = None
            return True
        except Exception:
            return False

    def autosave_tick(self):
        if (time.time() - self._last_save) >= self.autosave_interval:
            self.save()

    def save_on_exit(self):
        try:
            self.save()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # History & drift reporting
    def _summarize_snapshot(self, snap: Dict[str, Any], digest: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "timestamp": snap.get("timestamp", time.time()),
            "version": snap.get("version"),
            "hash": digest,
            "components": {},
        }
        components: Dict[str, Any] = {}
        for name, payload in snap.items():
            if name in ("timestamp", "version"):
                continue
            if isinstance(payload, dict):
                components[name] = {
                    "size": len(payload),
                    "keys_sample": sorted(list(payload.keys()))[:8],
                }
            else:
                components[name] = {"size": 0, "type": type(payload).__name__}
        summary["components"] = components
        return summary

    def _compute_drift(
        self,
        prev: Optional[Dict[str, Any]],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not prev:
            components = current.get("components", {})
            return {
                "severity": 1.0 if components else 0.0,
                "new": sorted(components.keys()),
                "removed": [],
                "deltas": {name: data.get("size", 0) for name, data in components.items()},
            }
        prev_components = prev.get("components", {})
        curr_components = current.get("components", {})
        deltas: Dict[str, int] = {}
        for name, info in curr_components.items():
            prev_size = int(prev_components.get(name, {}).get("size", 0))
            curr_size = int(info.get("size", 0))
            delta = curr_size - prev_size
            if delta:
                deltas[name] = delta
        removed = [name for name in prev_components.keys() if name not in curr_components]
        new = [name for name in curr_components.keys() if name not in prev_components]
        if not deltas and not new and not removed:
            severity = 0.0
        else:
            norm = max(len(curr_components) or 1, 1)
            severity = sum(abs(v) for v in deltas.values()) / norm
            if new or removed:
                severity = min(1.0, severity + 0.2)
        return {
            "severity": round(float(severity), 4),
            "new": sorted(new),
            "removed": sorted(removed),
            "deltas": deltas,
        }

    def _record_history(self, payload: bytes, summary: Dict[str, Any]) -> None:
        ts = summary.get("timestamp", time.time())
        try:
            stamp = datetime.utcfromtimestamp(float(ts)).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        digest = summary.get("hash", "snapshot")
        base = f"{stamp}_{digest[:12]}"
        data_path = os.path.join(self.history_dir, f"{base}.pkl")
        meta_path = os.path.join(self.history_dir, f"{base}.json")
        try:
            with open(data_path, "wb") as fh:
                fh.write(payload)
        except Exception:
            return
        meta_payload = dict(summary)
        meta_payload["path"] = os.path.relpath(data_path, self.directory)
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh, indent=2, sort_keys=True)
        except Exception:
            # History metadata is best-effort; ignore failures.
            pass

    def _prune_history(self) -> None:
        try:
            entries = [
                f
                for f in os.listdir(self.history_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(self.history_dir, f))
            ]
        except Exception:
            return
        if len(entries) <= self.history_retention:
            return
        entries.sort()
        obsolete = entries[: len(entries) - self.history_retention]
        for meta_name in obsolete:
            base, _ = os.path.splitext(meta_name)
            for suffix in (".json", ".pkl"):
                path = os.path.join(self.history_dir, f"{base}{suffix}")
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    def get_last_snapshot_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the latest persisted snapshot."""

        return dict(self._last_snapshot_meta or {})

    def get_last_drift(self) -> Dict[str, Any]:
        """Return the most recent drift analysis between snapshots."""

        return dict(self._last_drift or {})
