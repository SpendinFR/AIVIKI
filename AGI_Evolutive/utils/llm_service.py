"""Service layer to orchestrate repository-wide LLM integrations."""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from itertools import islice

from .llm_client import LLMCallError, LLMResult, OllamaLLMClient, OllamaModelConfig
from .llm_specs import LLMIntegrationSpec, get_spec
from AGI_Evolutive.core.config import cfg


class _PriorityToken:
    """Small RAII helper used by the priority controller."""

    __slots__ = ("_controller", "_is_user", "_released")

    def __init__(self, controller: "LLMPriorityController", is_user: bool) -> None:
        self._controller = controller
        self._is_user = is_user
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._controller._release(self._is_user)
            self._released = True

    def __enter__(self) -> "_PriorityToken":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.release()


class LLMPriorityController:
    """Serialises background calls while a user-focused interaction is active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._active_user_calls = 0

    def acquire(self, *, is_user: bool, timeout: float | None = None) -> _PriorityToken:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._cond:
            if is_user:
                self._active_user_calls += 1
                return _PriorityToken(self, True)

            while self._active_user_calls > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError("LLM busy with user-focused interaction")
                self._cond.wait(remaining)

            return _PriorityToken(self, False)

    def _release(self, is_user: bool) -> None:
        with self._cond:
            if is_user:
                self._active_user_calls = max(0, self._active_user_calls - 1)
                if self._active_user_calls == 0:
                    self._cond.notify_all()
            else:
                self._cond.notify_all()


_priority_controller = LLMPriorityController()
_user_focus_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agi_llm_user_focus", default=False
)


@contextmanager
def user_focus_scope():
    """Mark the current thread of execution as serving a user interaction."""

    token = _user_focus_flag.set(True)
    try:
        yield
    finally:
        _user_focus_flag.reset(token)


def _in_user_focus() -> bool:
    try:
        return bool(_user_focus_flag.get())
    except LookupError:  # pragma: no cover - defensive
        return False


def is_user_focus_active() -> bool:
    """Return True when the current execution context serves a user interaction."""

    return _in_user_focus()


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return default


_DEFAULT_ENABLED = _env_flag("AGI_ENABLE_LLM") and not _env_flag("AGI_DISABLE_LLM")


class LLMIntegrationError(RuntimeError):
    """Raised when the structured LLM integration fails."""


class LLMUnavailableError(LLMIntegrationError):
    """Raised when LLM integration is disabled or unavailable."""


@dataclass
class LLMInvocation:
    """Encapsulate the result of one integration call."""

    spec: LLMIntegrationSpec
    result: LLMResult


@dataclass(frozen=True)
class LLMCallRecord:
    """Trace the outcome of an integration attempt for diagnostics."""

    spec_key: str
    status: str
    timestamp: float
    message: Optional[str] = None


_ACTIVITY_LOG: deque[LLMCallRecord] = deque(maxlen=200)


def _record_activity(spec_key: str, status: str, message: Optional[str] = None) -> None:
    try:
        _ACTIVITY_LOG.appendleft(
            LLMCallRecord(
                spec_key=spec_key,
                status=status,
                timestamp=time.time(),
                message=message.strip() if isinstance(message, str) else message,
            )
        )
    except Exception:  # pragma: no cover - defensive guard for diagnostics
        pass


def get_recent_llm_activity(limit: int = 20) -> Sequence[LLMCallRecord]:
    """Return the most recent integration attempts for observability."""

    if limit is None or limit <= 0:
        limit = len(_ACTIVITY_LOG)
    return tuple(islice(_ACTIVITY_LOG, 0, limit))


class LLMIntegrationManager:
    """Coordinates prompt construction and model selection for integrations."""

    def __init__(
        self,
        *,
        client: Optional[OllamaLLMClient] = None,
        model_configs: Optional[Mapping[str, OllamaModelConfig]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._client = client or OllamaLLMClient()
        self._enabled = _DEFAULT_ENABLED if enabled is None else bool(enabled)
        self._model_configs: MutableMapping[str, OllamaModelConfig] = dict(model_configs or {})
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def call_json(
        self,
        spec_key: str,
        *,
        input_payload: Any | None = None,
        extra_instructions: Optional[Sequence[str]] = None,
        max_retries: int = 1,
    ) -> LLMInvocation:
        if not self._enabled:
            raise LLMUnavailableError("LLM integration is disabled")

        spec = get_spec(spec_key)
        instructions: list[str] = list(spec.extra_instructions)
        if extra_instructions:
            instructions.extend(instr.strip() for instr in extra_instructions if instr and instr.strip())
        instructions.append("Si tu n'es pas certain, explique l'incertitude dans le champ 'notes'.")

        is_user_focus = _in_user_focus()
        try:
            timeout = None if is_user_focus else float(cfg().get("LLM_BACKGROUND_WAIT_TIMEOUT", 45.0))
        except Exception:
            timeout = None if is_user_focus else 45.0

        try:
            with _priority_controller.acquire(is_user=is_user_focus, timeout=timeout):
                result = self._client.generate_json(
                    self._resolve_model(spec.preferred_model),
                    spec.prompt_goal,
                    input_data=input_payload,
                    extra_instructions=instructions,
                    example_output=spec.example_output,
                    max_retries=max_retries,
                )
        except TimeoutError as exc:
            raise LLMIntegrationError(
                "LLM call skipped: background request waited too long during user focus"
            ) from exc
        except LLMCallError as exc:  # pragma: no cover - delegated to integration error
            raise LLMIntegrationError(f"LLM call failed for spec '{spec_key}': {exc}") from exc

        return LLMInvocation(spec=spec, result=result)

    def call_dict(
        self,
        spec_key: str,
        *,
        input_payload: Any | None = None,
        extra_instructions: Optional[Sequence[str]] = None,
        max_retries: int = 1,
    ) -> Mapping[str, Any]:
        invocation = self.call_json(
            spec_key,
            input_payload=input_payload,
            extra_instructions=extra_instructions,
            max_retries=max_retries,
        )
        parsed = invocation.result.parsed
        if not isinstance(parsed, Mapping):
            raise LLMIntegrationError(
                f"Spec '{spec_key}' returned a non-mapping payload: {type(parsed).__name__}"
            )
        return parsed

    def _resolve_model(self, model_name: str) -> OllamaModelConfig:
        with self._lock:
            if model_name not in self._model_configs:
                self._model_configs[model_name] = OllamaModelConfig(name=model_name)
            return self._model_configs[model_name]


_default_manager: Optional[LLMIntegrationManager] = None
_default_lock = threading.Lock()


def get_llm_manager() -> LLMIntegrationManager:
    global _default_manager
    with _default_lock:
        if _default_manager is None:
            _default_manager = LLMIntegrationManager()
        return _default_manager


def set_llm_manager(manager: Optional[LLMIntegrationManager]) -> None:
    global _default_manager
    with _default_lock:
        _default_manager = manager


def is_llm_enabled() -> bool:
    manager = _default_manager
    if manager is not None:
        return manager.enabled
    return _DEFAULT_ENABLED


def try_call_llm_dict(
    spec_key: str,
    *,
    input_payload: Any | None = None,
    extra_instructions: Optional[Sequence[str]] = None,
    logger: Optional[Any] = None,
    max_retries: int = 1,
) -> Optional[Mapping[str, Any]]:
    """Attempt to call the shared LLM and return a mapping payload.

    The helper centralises the common guard/exception handling logic used by the
    individual modules so they can focus on crafting the structured payload
    without duplicating availability checks.  When the LLM is disabled or the
    integration fails, ``None`` is returned to signal the caller to fall back to
    heuristics.
    """

    if not is_llm_enabled():
        _record_activity(spec_key, "disabled", "LLM integration désactivée")
        return None

    try:
        manager = get_llm_manager()
        payload = manager.call_dict(
            spec_key,
            input_payload=input_payload,
            extra_instructions=extra_instructions,
            max_retries=max_retries,
        )
        _record_activity(spec_key, "success", None)
        return payload
    except (LLMUnavailableError, LLMIntegrationError) as exc:
        _record_activity(spec_key, "error", str(exc))
        if logger is not None:
            try:
                logger.debug(
                    "LLM integration '%s' unavailable: %s", spec_key, exc, exc_info=True
                )
            except Exception:  # pragma: no cover - defensive logging guard
                pass
        return None
    except Exception as exc:  # pragma: no cover - unexpected failure safety net
        _record_activity(spec_key, "error", str(exc))
        if logger is not None:
            try:
                logger.warning(
                    "Unexpected error while calling LLM integration '%s': %s",
                    spec_key,
                    exc,
                    exc_info=True,
                )
            except Exception:
                pass
        return None


__all__ = [
    "LLMIntegrationError",
    "LLMIntegrationManager",
    "LLMInvocation",
    "LLMCallRecord",
    "LLMUnavailableError",
    "LLMPriorityController",
    "get_llm_manager",
    "get_recent_llm_activity",
    "is_llm_enabled",
    "set_llm_manager",
    "try_call_llm_dict",
    "user_focus_scope",
]
