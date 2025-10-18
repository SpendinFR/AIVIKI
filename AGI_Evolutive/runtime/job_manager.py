# Gestionnaire de jobs complet (priorités, deux files, budgets, annulation,
# idempotence, progrès, persistance JSONL, drain des complétions côté thread principal).
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, List, Deque, Tuple
import time, threading, heapq, os, json, uuid, traceback, collections


def _now() -> float:
    return time.time()


@dataclass(order=True)
class _PQItem:
    # heapq ordonne par (neg_prio, created, seq, job_id)
    sort_key: Tuple[float, float, int] = field(init=False, repr=False)
    neg_prio: float
    created_ts: float
    seq: int
    job_id: str

    def __post_init__(self):
        self.sort_key = (self.neg_prio, self.created_ts, self.seq)


@dataclass
class Job:
    id: str
    kind: str  # ex: "io", "compute", "nlp"
    queue: str  # "interactive" | "background"
    priority: float  # 0..1
    fn: Optional[Callable] = field(repr=False, default=None)
    args: Dict[str, Any] = field(default_factory=dict)
    key: Optional[str] = None  # idempotence key (même job)
    timeout_s: Optional[float] = None
    status: str = "queued"  # queued|running|done|error|cancelled
    created_ts: float = field(default_factory=_now)
    started_ts: float = 0.0
    finished_ts: float = 0.0
    progress: float = 0.0  # 0..1
    result: Any = None
    error: Optional[str] = None
    trace: Optional[str] = None


class JobContext:
    """Contexte passé à la fonction du job (pour progrès, cancel, timeouts)."""

    def __init__(self, jm: "JobManager", job_id: str):
        self._jm = jm
        self._job_id = job_id

    def update_progress(self, p: float):
        self._jm._update_progress(self._job_id, max(0.0, min(1.0, float(p))))

    def cancelled(self) -> bool:
        return self._jm._is_cancelled(self._job_id)


class JobManager:
    """
    Deux files : interactive (latence faible) et background (batch).
    Priorités [0..1], budgets par tick, idempotence, annulation, timeouts,
    persistance JSONL, complétions drainables par le thread principal.
    """

    def __init__(self, arch, data_dir: str = "data", workers_interactive: int = 1, workers_background: int = 2):
        self.arch = arch
        self.data_dir = data_dir
        self.paths = {
            "log": os.path.join(self.data_dir, "runtime/jobs.log.jsonl"),
            "state": os.path.join(self.data_dir, "runtime/jobs.state.json"),
        }
        os.makedirs(os.path.dirname(self.paths["log"]), exist_ok=True)

        self._lock = threading.RLock()
        self._seq = 0
        self._jobs: Dict[str, Job] = {}
        self._key_map: Dict[str, str] = {}  # key -> job_id (idempotence)
        self._cancelled: set[str] = set()
        self._pq_inter: List[_PQItem] = []
        self._pq_back: List[_PQItem] = []
        self._completions: Deque[Dict[str, Any]] = collections.deque(maxlen=512)

        # budgets (ajuste si besoin)
        self.budgets = {
            "interactive": {"max_running": workers_interactive},
            "background": {"max_running": workers_background},
        }
        self._running_inter = 0
        self._running_back = 0

        # Démarre les workers
        self._alive = True
        self._workers: List[threading.Thread] = []
        for _ in range(workers_interactive):
            t = threading.Thread(target=self._worker_loop, args=("interactive",), daemon=True)
            t.start()
            self._workers.append(t)
        for _ in range(workers_background):
            t = threading.Thread(target=self._worker_loop, args=("background",), daemon=True)
            t.start()
            self._workers.append(t)

    # ------------------- API publique -------------------
    def submit(
        self,
        *,
        kind: str,
        fn: Callable[[JobContext, Dict[str, Any]], Any],
        args: Dict[str, Any] | None = None,
        queue: str = "background",
        priority: float = 0.5,
        key: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> str:
        """Dépose un job. Idempotent si `key` est fournie."""
        args = args or {}
        priority = max(0.0, min(1.0, float(priority)))
        queue = "interactive" if queue == "interactive" else "background"

        with self._lock:
            if key and key in self._key_map:
                jid = self._key_map[key]
                j = self._jobs.get(jid)
                if j and j.status in {"queued", "running"}:
                    return jid

            jid = str(uuid.uuid4())
            job = Job(
                id=jid,
                kind=kind,
                queue=queue,
                priority=priority,
                fn=fn,
                args=args,
                key=key,
                timeout_s=timeout_s,
            )
            self._jobs[jid] = job
            if key:
                self._key_map[key] = jid
            self._push_pq(job)
            self._log({"event": "submit", "job": self._job_view(job)})
            return jid

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs and self._jobs[job_id].status in {"queued", "running"}:
                self._cancelled.add(job_id)
                self._log({"event": "cancel", "job_id": job_id})
                return True
            return False

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            return self._job_view(j) if j else None

    def poll_completed(self, max_n: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for _ in range(min(max_n, len(self._completions))):
                out.append(self._completions.popleft())
        return out

    def drain_to_memory(self, memory) -> int:
        """A appeler côté thread principal (ex: dans _tick_background_systems)."""
        done = self.poll_completed(128)
        n = 0
        for ev in done:
            try:
                memory.add_memory({"kind": "job_event", "content": ev.get("event", ""), "metadata": ev})
                n += 1
            except Exception:
                pass
        return n

    # ------------------- internes -------------------
    def _push_pq(self, job: Job):
        self._seq += 1
        item = _PQItem(neg_prio=-float(job.priority), created_ts=job.created_ts, seq=self._seq, job_id=job.id)
        if job.queue == "interactive":
            heapq.heappush(self._pq_inter, item)
        else:
            heapq.heappush(self._pq_back, item)

    def _pop_next(self, lane: str) -> Optional[str]:
        with self._lock:
            pq = self._pq_inter if lane == "interactive" else self._pq_back
            if not pq:
                return None
            # budget: nombre max de jobs simultanés par file
            running = self._running_inter if lane == "interactive" else self._running_back
            maxrun = self.budgets[lane]["max_running"]
            if running >= maxrun:
                return None
            item = heapq.heappop(pq)
            return item.job_id

    def _start_job(self, j: Job):
        j.status = "running"
        j.started_ts = _now()
        if j.queue == "interactive":
            self._running_inter += 1
        else:
            self._running_back += 1
        self._log({"event": "start", "job": self._job_view(j)})

    def _finish_job(self, j: Job, ok: bool, result: Any = None, error: Optional[str] = None, trace: Optional[str] = None):
        j.finished_ts = _now()
        j.status = "done" if ok else ("cancelled" if (j.id in self._cancelled) else "error")
        j.result = result
        j.error = error
        j.trace = trace
        if j.queue == "interactive":
            self._running_inter = max(0, self._running_inter - 1)
        else:
            self._running_back = max(0, self._running_back - 1)
        ev = {"event": j.status, "job": self._job_view(j)}
        with self._lock:
            self._completions.append(ev)
        self._log(ev)
        self._persist_state()

    def _update_progress(self, job_id: str, p: float):
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return
            j.progress = p
            self._log({"event": "progress", "job_id": job_id, "progress": float(p)})

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _worker_loop(self, lane: str):
        while self._alive:
            jid = self._pop_next(lane)
            if not jid:
                time.sleep(0.01)
                continue
            with self._lock:
                j = self._jobs.get(jid)
            if not j:
                continue
            # cancellation avant start ?
            if self._is_cancelled(j.id):
                self._finish_job(j, ok=False, error="cancelled_before_start")
                continue
            self._start_job(j)
            ctx = JobContext(self, j.id)
            ok, result, err, tr = True, None, None, None
            try:
                # timeout soft: on laisse la fonction vérifier ctx.cancelled() périodiquement
                result = j.fn(ctx, j.args or {})
            except Exception as e:
                ok, err = False, str(e)
                tr = traceback.format_exc()
            # cancellation après exécution ?
            if self._is_cancelled(j.id) and ok:
                ok, err = False, "cancelled"
            self._finish_job(j, ok=ok, result=result, error=err, trace=tr)

    def _job_view(self, j: Optional[Job]) -> Optional[Dict[str, Any]]:
        if not j:
            return None
        return {
            "id": j.id,
            "kind": j.kind,
            "queue": j.queue,
            "priority": j.priority,
            "status": j.status,
            "progress": j.progress,
            "created_ts": j.created_ts,
            "started_ts": j.started_ts,
            "finished_ts": j.finished_ts,
            "timeout_s": j.timeout_s,
            "key": j.key,
        }

    def _log(self, obj: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(self.paths["log"]), exist_ok=True)
            with open(self.paths["log"], "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _persist_state(self):
        try:
            state = {jid: self._job_view(j) for jid, j in self._jobs.items()}
            with open(self.paths["state"], "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def snapshot_identity_view(self) -> Dict[str, Any]:
        """Small summary of running and recent jobs for the SelfModel."""
        now = time.time()
        view: Dict[str, Any] = {
            "current": {
                "focus_topic": None,
                "jobs_running": [],
                "loads": {"interactive": 0, "background": 0},
            },
            "recent": [],
        }
        try:
            with self._lock:
                jobs = list(self._jobs.values())
                pq_inter = len(self._pq_inter)
                pq_back = len(self._pq_back)
                running_inter = self._running_inter
                running_back = self._running_back

            arch = getattr(self, "arch", None)
            focus_topic = None
            if arch is not None:
                focus_topic = (
                    getattr(arch, "_current_topic", None)
                    or getattr(arch, "current_topic", None)
                )
            view["current"]["focus_topic"] = focus_topic

            running_jobs = [j for j in jobs if j.status == "running"]
            view["current"]["jobs_running"] = [
                {
                    "id": j.id,
                    "kind": j.kind,
                    "since": j.started_ts or j.created_ts,
                }
                for j in running_jobs
            ]

            view["current"]["loads"]["interactive"] = running_inter + pq_inter
            view["current"]["loads"]["background"] = running_back + pq_back

            completed = [
                j
                for j in jobs
                if j.status in {"done", "error", "cancelled"}
            ]
            completed.sort(
                key=lambda j: j.finished_ts or j.created_ts,
                reverse=True,
            )
            for job in completed[:50]:
                finished_ts = job.finished_ts or job.created_ts or now
                created_ts = job.created_ts or finished_ts
                latency_ms = int(max(0.0, (finished_ts - created_ts)) * 1000.0)
                status = job.status
                if status == "cancelled":
                    status = "error"
                view["recent"].append(
                    {
                        "job_id": job.id,
                        "kind": job.kind,
                        "status": status,
                        "ts": finished_ts,
                        "latency_ms": latency_ms,
                    }
                )

            log_path = self.paths.get("log")
            if log_path and os.path.exists(log_path):
                added = {
                    (entry.get("job_id"), entry.get("ts"))
                    for entry in view["recent"]
                    if entry.get("job_id") is not None
                }
                with open(log_path, "r", encoding="utf-8") as handle:
                    tail = handle.readlines()[-500:]
                for line in tail:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if event.get("event") not in {"job_done", "job_error"}:
                        continue
                    job_id = event.get("job_id")
                    ts = float(event.get("ts", now))
                    key = (job_id, ts)
                    if key in added:
                        continue
                    status = "done" if event.get("event") == "job_done" else "error"
                    latency_ms = int(float(event.get("latency", 0.0)) * 1000.0)
                    view["recent"].append(
                        {
                            "job_id": job_id,
                            "kind": event.get("kind"),
                            "status": status,
                            "ts": ts,
                            "latency_ms": latency_ms,
                        }
                    )
                    added.add(key)

            view["recent"].sort(key=lambda entry: entry.get("ts", 0.0), reverse=True)
            view["recent"] = view["recent"][:200]
        except Exception:
            pass
        return view
