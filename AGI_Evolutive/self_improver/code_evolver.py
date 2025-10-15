from __future__ import annotations

import ast
import contextlib
import difflib
import importlib
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .quality import QualityGateRunner


@dataclass
class CodePatch:
    patch_id: str
    target_file: Path
    module_name: str
    original_source: str
    patched_source: str
    summary: str


class _ScoreHeuristicTweaker(ast.NodeTransformer):
    """Mutate numeric constants in the abduction score heuristic."""

    def __init__(self) -> None:
        self._context: List[str] = []
        self._changed: bool = False
        self._metadata: Dict[str, Any] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._context.append(node.name)
        new_node = self.generic_visit(node)
        self._context.pop()
        return new_node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # type: ignore[override]
        if self._changed:
            return node
        if not self._context or self._context[-1] != "_score":
            return node
        if not isinstance(node.value, (int, float)):
            return node
        original = float(node.value)
        if original <= 0.0 or original > 2.0:
            return node
        delta = random.uniform(-0.08, 0.08)
        candidate = max(0.0, min(2.0, original + delta))
        if candidate == original:
            return node
        self._changed = True
        rounded = round(candidate, 3)
        self._metadata = {
            "kind": "heuristic_weight_shift",
            "from": original,
            "to": rounded,
            "delta": round(rounded - original, 3),
        }
        return ast.copy_location(ast.Constant(value=rounded), node)


class CodeEvolver:
    """Propose and evaluate AST-level patches for the cognitive architecture."""

    def __init__(
        self,
        repo_root: str,
        sandbox: Any,
        quality: QualityGateRunner,
        arch_factory: Any,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.sandbox = sandbox
        self.quality = quality
        self.arch_factory = arch_factory
        self._targets: List[Dict[str, str]] = [
            {
                "file": "reasoning/abduction.py",
                "module": "AGI_Evolutive.reasoning.abduction",
            }
        ]

    # ------------------------------------------------------------------
    def _load_source(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _patch_source(self, source: str) -> Optional[CodePatch]:
        tree = ast.parse(source)
        mutator = _ScoreHeuristicTweaker()
        patched = mutator.visit(tree)
        if not mutator.metadata:
            return None
        ast.fix_missing_locations(patched)
        patched_source = ast.unparse(patched)
        summary = (
            "Ajustement automatique d'un poids heuristique dans abduction._score"
        )
        patch = CodePatch(
            patch_id=str(uuid.uuid4()),
            target_file=Path(),  # placeholder
            module_name="",
            original_source=source,
            patched_source=patched_source,
            summary=summary,
        )
        # Attach metadata onto object for reporting
        setattr(patch, "_metadata", mutator.metadata)
        return patch

    def _prepare_patch(self, target: Dict[str, str]) -> Optional[CodePatch]:
        path = self.repo_root / target["file"]
        if not path.exists():
            return None
        source = self._load_source(path)
        candidate = self._patch_source(source)
        if not candidate:
            return None
        candidate.target_file = path
        candidate.module_name = target["module"]
        return candidate

    def generate_candidates(self, n: int = 2) -> List[CodePatch]:
        candidates: List[CodePatch] = []
        for _ in range(max(1, n)):
            target = random.choice(self._targets)
            patch = self._prepare_patch(target)
            if patch:
                candidates.append(patch)
        return candidates

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _apply_patch(self, patch: CodePatch) -> Iterable[None]:
        path = patch.target_file
        backup = self._load_source(path)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(patch.patched_source)
            importlib.invalidate_caches()
            module = importlib.import_module(patch.module_name)
            importlib.reload(module)
            yield
        finally:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(backup)
            importlib.invalidate_caches()
            try:
                module = importlib.import_module(patch.module_name)
                importlib.reload(module)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _lint_patch(self, patch: CodePatch) -> Dict[str, Any]:
        temp_dir = tempfile.mkdtemp(prefix="code_evolver_")
        temp_file = Path(temp_dir) / patch.target_file.name
        with open(temp_file, "w", encoding="utf-8") as handle:
            handle.write(patch.patched_source)
        errors: List[str] = []
        try:
            compile(patch.patched_source, str(temp_file), "exec")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return {"passed": not errors, "errors": errors}

    def _static_analysis(self, patch: CodePatch) -> Dict[str, Any]:
        tree = ast.parse(patch.patched_source)
        banned = {ast.Exec, ast.Global, ast.Nonlocal}
        violations: List[str] = []
        for node in ast.walk(tree):
            if type(node) in banned:  # noqa: E721 - intentional exact type check
                violations.append(type(node).__name__)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    violations.append(f"call:{node.func.id}")
        return {"passed": not violations, "violations": violations}

    def _build_diff(self, patch: CodePatch) -> str:
        original_lines = patch.original_source.splitlines(keepends=True)
        patched_lines = patch.patched_source.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=str(patch.target_file),
            tofile=f"{patch.target_file} (patched)",
        )
        return "".join(diff)

    # ------------------------------------------------------------------
    def evaluate_patch(
        self,
        patch: CodePatch,
        baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        lint_report = self._lint_patch(patch)
        static_report = self._static_analysis(patch)
        if not lint_report["passed"] or not static_report["passed"]:
            return {
                "passed": False,
                "lint": lint_report,
                "static": static_report,
            }

        with self._apply_patch(patch):
            quality_report = self.quality.run({})
            evaluation = self.sandbox.run_all({})
            canary = self.sandbox.run_canary({}, baseline_metrics or {})

        diff = self._build_diff(patch)
        report = {
            "passed": quality_report["passed"]
            and evaluation.get("security", {}).get("passed", True)
            and canary.get("passed", False),
            "lint": lint_report,
            "static": static_report,
            "quality": quality_report,
            "evaluation": evaluation,
            "canary": canary,
            "diff": diff,
            "summary": patch.summary,
            "metadata": getattr(patch, "_metadata", {}),
            "module": patch.module_name,
            "file": str(patch.target_file.relative_to(self.repo_root)),
        }
        return report

    # ------------------------------------------------------------------
    def promote_patch(self, patch_payload: Dict[str, Any]) -> None:
        file_rel = patch_payload.get("file")
        if not file_rel:
            raise ValueError("Missing file information for patch promotion")
        path = self.repo_root / file_rel
        patched_source = patch_payload.get("patched_source")
        if not patched_source:
            raise ValueError("Missing patched source for promotion")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(patched_source)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(patch_payload.get("module", ""))
            importlib.reload(module)
        except Exception:
            pass

    def serialise_patch(self, patch: CodePatch) -> Dict[str, Any]:
        return {
            "patch_id": patch.patch_id,
            "file": str(patch.target_file.relative_to(self.repo_root)),
            "module": patch.module_name,
            "patched_source": patch.patched_source,
            "summary": patch.summary,
            "metadata": getattr(patch, "_metadata", {}),
            "diff": self._build_diff(patch),
        }
