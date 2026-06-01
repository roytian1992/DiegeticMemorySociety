from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dms.intent_levels import normalize_prompt_id

_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class PromptSpec:
    """A YAML prompt template plus declared render variables."""

    id: str
    template: str
    task_variables: list[dict[str, Any]]
    static_variables: list[dict[str, Any]]
    raw: dict[str, Any]
    path: Path


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _declared_names(variables: Any) -> set[str]:
    if not isinstance(variables, list):
        return set()
    names: set[str] = set()
    for item in variables:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            name = item["name"].strip()
            if name:
                names.add(name)
    return names


def _required_names(variables: Any) -> set[str]:
    if not isinstance(variables, list):
        return set()
    names: set[str] = set()
    for item in variables:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip() and item.get("required") is True:
            names.add(name.strip())
    return names


def _safe_replace(template: str, declared: set[str], values: dict[str, Any]) -> str:
    rendered = {key: _stringify(value) for key, value in values.items() if key in declared}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in declared:
            return rendered.get(name, "")
        return match.group(0)

    return _VAR_PATTERN.sub(repl, template)


class YAMLPromptLoader:
    """Load YAML prompts by nested id and safely render declared variables.

    This follows the prompt-management style used in NarrativeKnowledgeWeaver:
    prompts live as YAML files under a prompt root, each file declares
    task_variables and static_variables, and rendering only replaces declared
    placeholders.
    """

    def __init__(self, prompt_dir: str | Path, global_static: dict[str, Any] | None = None):
        self.prompt_dir = Path(prompt_dir)
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt dir not found: {self.prompt_dir}")
        self.global_static = global_static or {}

    def load(self, prompt_id: str) -> PromptSpec:
        pid = (prompt_id or "").strip()
        if not pid:
            raise ValueError("prompt_id must be a non-empty string")

        direct = Path(pid)
        if direct.is_absolute() and direct.is_file():
            return self._load_file(direct)

        pid = self._strip_prompt_dir_prefix(pid)
        pid = normalize_prompt_id(pid)
        if pid.endswith((".yaml", ".yml")):
            path = self.prompt_dir / pid
            if not path.is_file():
                raise FileNotFoundError(f"Prompt yaml not found: {path}")
            return self._load_file(path)

        if "/" in pid or "\\" in pid:
            rel = pid.replace("\\", "/")
            for suffix in (".yaml", ".yml"):
                path = self.prompt_dir / f"{rel}{suffix}"
                if path.is_file():
                    return self._load_file(path)
            raise FileNotFoundError(f"Prompt yaml not found for id: {prompt_id}")

        candidates = list(self.prompt_dir.rglob(f"{pid}.yaml")) + list(self.prompt_dir.rglob(f"{pid}.yml"))
        if not candidates:
            raise FileNotFoundError(f"Prompt yaml id not found: {prompt_id}")
        if len(candidates) > 1:
            joined = "\n".join(str(path) for path in sorted(candidates))
            raise RuntimeError(f"Ambiguous prompt_id '{prompt_id}'. Multiple matches:\n{joined}")
        return self._load_file(candidates[0])

    def render(
        self,
        prompt: str | PromptSpec,
        *,
        task_values: dict[str, Any] | None = None,
        static_values: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> str:
        spec = self.load(prompt) if isinstance(prompt, str) else prompt
        task_values = task_values or {}
        static_values = static_values or {}

        declared = _declared_names(spec.task_variables) | _declared_names(spec.static_variables)
        required_task = _required_names(spec.task_variables)
        required_static = _required_names(spec.static_variables)

        values: dict[str, Any] = {}
        values.update(self.global_static)
        values.update(static_values)
        values.update(task_values)

        if strict:
            missing_task = sorted(name for name in required_task if name not in values)
            missing_static = sorted(name for name in required_static if name not in values)
            if missing_task or missing_static:
                raise ValueError(f"Missing vars: static={missing_static} task={missing_task}")

        return _safe_replace(spec.template, declared, values)

    def _load_file(self, path: Path) -> PromptSpec:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid prompt yaml, expected object: {path}")

        template = payload.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Missing template in prompt yaml: {path}")

        task_variables = payload.get("task_variables") or []
        static_variables = payload.get("static_variables") or []
        if not isinstance(task_variables, list):
            raise ValueError(f"task_variables must be a list in {path}")
        if not isinstance(static_variables, list):
            raise ValueError(f"static_variables must be a list in {path}")

        return PromptSpec(
            id=str(payload.get("id") or path.stem).strip(),
            template=template,
            task_variables=task_variables,
            static_variables=static_variables,
            raw=payload,
            path=path,
        )

    def _strip_prompt_dir_prefix(self, prompt_id: str) -> str:
        try:
            prompt_root = self.prompt_dir.resolve()
            prompt_path = Path(prompt_id).resolve()
            if str(prompt_path).startswith(str(prompt_root) + str(Path.sep)):
                return prompt_path.relative_to(prompt_root).as_posix()
        except Exception:
            pass

        normalized_root = str(self.prompt_dir).replace("\\", "/").rstrip("/")
        normalized_id = prompt_id.replace("\\", "/")
        if normalized_id.startswith(normalized_root + "/"):
            return normalized_id[len(normalized_root) + 1 :]
        return prompt_id
