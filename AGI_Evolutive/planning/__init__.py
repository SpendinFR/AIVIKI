"""Planning helpers for HTN-style task decomposition."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict

from .htn import HTNPlanner

LOGGER = logging.getLogger(__name__)


def _normalize_goal(goal: Any) -> str:
    text = str(goal or "").strip()
    return text


def _sanitize_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in context.items():
        try:
            sanitized[str(key)] = json_sanitize(value)
        except Exception:
            continue
    return sanitized


def _coerce_str_list(values: Any) -> List[str]:
    items: List[str] = []
    if isinstance(values, Mapping):
        values = values.values()
    if isinstance(values, (str, bytes)):
        values = [values]
    if isinstance(values, Iterable):
        for value in values:
            if isinstance(value, (str, bytes)):
                cleaned = str(value).strip()
                if cleaned:
                    items.append(cleaned)
    return items


def _clamp_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:  # NaN guard
        return default
    return max(0.0, min(1.0, numeric))


def _coerce_steps(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, Mapping):
        data = [data]
    steps: List[Dict[str, Any]] = []
    if not isinstance(data, Iterable):
        return steps

    for index, raw_step in enumerate(data, start=1):
        if isinstance(raw_step, Mapping):
            description = str(raw_step.get("description") or raw_step.get("text") or raw_step.get("action") or "").strip()
            if not description:
                continue
            step_id = str(raw_step.get("id") or raw_step.get("name") or f"step_{index}").strip()
            if not step_id:
                step_id = f"step_{index}"
            depends_on = _coerce_str_list(
                raw_step.get("depends_on")
                or raw_step.get("dependencies")
                or raw_step.get("requires")
                or []
            )
            priority = raw_step.get("priority")
            try:
                priority_value = int(priority)
            except (TypeError, ValueError):
                priority_value = index
            steps.append(
                {
                    "id": step_id,
                    "description": description,
                    "depends_on": depends_on,
                    "priority": max(1, priority_value),
                }
            )
        elif isinstance(raw_step, (str, bytes)):
            description = str(raw_step).strip()
            if description:
                steps.append(
                    {
                        "id": f"step_{index}",
                        "description": description,
                        "depends_on": [],
                        "priority": index,
                    }
                )
    return steps


def _steps_to_text(steps: Sequence[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    for index, step in enumerate(steps, start=1):
        description = str(step.get("description", "")).strip()
        if not description:
            continue
        prefix = f"{index}. "
        depends_on = _coerce_str_list(step.get("depends_on", []))
        if depends_on:
            suffix = f" (dépend de: {', '.join(depends_on)})"
        else:
            suffix = ""
        lines.append(f"{prefix}{description}{suffix}")
    return lines


def _collect_templates(planner: Optional[HTNPlanner]) -> List[str]:
    if planner is None:
        return []
    templates = getattr(planner, "_templates", None)
    if not isinstance(templates, MutableMapping):
        return []
    names: List[str] = []
    for key in templates.keys():
        try:
            names.append(str(key))
        except Exception:
            continue
    return sorted({name for name in names if name})


def _compute_fallback_plan(
    planner: Optional[HTNPlanner], goal: str, context: Mapping[str, Any]
) -> List[str]:
    if planner is None or not goal:
        return []
    try:
        base_plan = super(HTNPlanner, planner).plan(goal, context=dict(context))
    except Exception:
        return []
    if not isinstance(base_plan, list):
        return []
    fallback: List[str] = []
    for item in base_plan:
        if isinstance(item, (str, bytes)):
            cleaned = str(item).strip()
            if cleaned:
                fallback.append(cleaned)
    return fallback


def _parse_llm_response(response: Optional[Mapping[str, Any]], goal: str) -> Optional[Dict[str, Any]]:
    if not isinstance(response, Mapping):
        return None

    plan_name = str(
        response.get("plan_name")
        or response.get("title")
        or response.get("objective")
        or goal
    ).strip() or goal

    steps = _coerce_steps(
        response.get("steps")
        or response.get("plan")
        or response.get("tasks")
    )

    text_steps = _coerce_str_list(response.get("text_steps"))
    if not text_steps and steps:
        text_steps = _steps_to_text(steps)

    risks = _coerce_str_list(response.get("risks") or response.get("dangers") or response.get("alerts"))

    notes_raw = response.get("notes") or response.get("comment")
    notes = str(notes_raw).strip() if isinstance(notes_raw, (str, bytes)) else ""

    return {
        "plan_name": plan_name,
        "steps": steps,
        "text_steps": text_steps,
        "risks": risks,
        "confidence": _clamp_float(response.get("confidence"), default=0.0),
        "notes": notes,
        "fallback_used": bool(response.get("fallback_used") or response.get("use_fallback")),
        "raw_response": json_sanitize(response),
    }


def _build_fallback_result(
    goal: str,
    fallback_plan: Sequence[str],
    *,
    notes: str = "",
    raw_response: Optional[Mapping[str, Any]] = None,
    reason: str = "fallback",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    plan_label = goal or "plan"
    text_steps = [str(item).strip() for item in fallback_plan if str(item).strip()]
    structured_steps = _coerce_steps(text_steps)
    result: Dict[str, Any] = {
        "plan_name": plan_label,
        "steps": structured_steps,
        "text_steps": text_steps,
        "risks": [],
        "confidence": _clamp_float(confidence, default=0.0) if confidence is not None else 0.0,
        "notes": notes,
        "source": "heuristic",
        "fallback_plan": text_steps,
        "fallback_used": True,
    }
    if raw_response is not None:
        result["raw_response"] = raw_response
    if reason:
        result["fallback_reason"] = reason
    return result


def generate_structured_plan(
    goal: Any,
    *,
    context: Optional[Mapping[str, Any]] = None,
    planner: Optional[HTNPlanner] = None,
) -> Dict[str, Any]:
    """Generate a structured plan using the LLM with heuristic fallback."""

    normalized_goal = _normalize_goal(goal)
    safe_context = _sanitize_context(context)
    fallback_plan = _compute_fallback_plan(planner, normalized_goal, safe_context)

    payload = {
        "goal": normalized_goal,
        "context": safe_context,
        "heuristic_plan": fallback_plan,
        "available_templates": _collect_templates(planner),
    }

    response = try_call_llm_dict(
        "planning_overview",
        input_payload=payload,
        logger=LOGGER,
    )

    parsed = _parse_llm_response(response, normalized_goal)

    if parsed and parsed.get("fallback_used") and fallback_plan:
        return _build_fallback_result(
            normalized_goal,
            fallback_plan,
            notes=parsed.get("notes", ""),
            raw_response=parsed.get("raw_response"),
            reason="llm_requested_fallback",
            confidence=parsed.get("confidence"),
        )

    if parsed and parsed.get("steps"):
        parsed_plan = dict(parsed)
        parsed_plan.setdefault("text_steps", _steps_to_text(parsed_plan["steps"]))
        parsed_plan.setdefault("risks", [])
        parsed_plan.setdefault("notes", "")
        parsed_plan["source"] = "llm"
        parsed_plan["fallback_plan"] = fallback_plan
        parsed_plan["fallback_used"] = bool(parsed_plan.get("fallback_used"))
        if "raw_response" not in parsed_plan and response is not None:
            parsed_plan["raw_response"] = json_sanitize(response)
        return parsed_plan

    # No LLM output or unusable response -> fallback to heuristics
    return _build_fallback_result(
        normalized_goal,
        fallback_plan,
        notes=parsed.get("notes", "") if parsed else "",
        raw_response=parsed.get("raw_response") if parsed else None,
        reason="llm_unavailable",
        confidence=parsed.get("confidence") if parsed else None,
    )


__all__ = ["HTNPlanner", "generate_structured_plan"]

