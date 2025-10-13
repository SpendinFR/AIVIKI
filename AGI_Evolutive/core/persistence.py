
# core/persistence.py
"""
PersistenceManager: sauvegarde/chargement robuste de l'état de l'AGI en local.
- Sauvegarde automatique à intervalle régulier et sur demande
- Reprise à chaud: si un snapshot existe, on recharge lors de l'initialisation
- Sérialisation prudente: on filtre les objets non-sérialisables
"""
import os, time, pickle, types, inspect
from typing import Any, Dict

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
        # ignorer méthodes, modules, fonctions, générateurs, coroutines...
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
        tmpfile = self.filename + ".tmp"
        with open(tmpfile, "wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmpfile, self.filename)
        self._last_save = time.time()
        return self.filename
    
    def load(self) -> bool:
        if not os.path.exists(self.filename):
            return False
        try:
            with open(self.filename, "rb") as f:
                snap = pickle.load(f)
            for name, state in snap.items():
                if name in ("timestamp", "version"): 
                    continue
                comp = getattr(self.arch, name, None)
                if comp is not None and isinstance(state, dict):
                    _from_state(comp, state)
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
