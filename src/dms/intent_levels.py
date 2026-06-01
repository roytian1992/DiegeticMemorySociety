from __future__ import annotations

CANONICAL_INTENT_LEVELS = (
    "social_simulation_intent",
    "writing_intent",
    "writing_spec",
)

LEGACY_INTENT_LEVEL_ALIASES = {
    "sparse": "social_simulation_intent",
    "brief": "social_simulation_intent",
    "social": "social_simulation_intent",
    "social_seed": "social_simulation_intent",
    "simulation": "social_simulation_intent",
    "simulation_intent": "social_simulation_intent",
    "author_intent": "writing_intent",
    "detailed": "writing_spec",
    "reference": "writing_spec",
    "eval": "writing_spec",
    "evaluation": "writing_spec",
    "reference_scene_spec": "writing_spec",
}

CLI_INTENT_LEVEL_CHOICES = (
    *CANONICAL_INTENT_LEVELS,
    "reference_scene_spec",
    "sparse",
    "detailed",
)

PROMPT_ID_ALIASES = {
    "reference_scene_spec": "writing_spec",
    "reference_scene_spec.yaml": "writing_spec.yaml",
    "dms/reference_scene_spec": "dms/writing_spec",
    "dms/reference_scene_spec.yaml": "dms/writing_spec.yaml",
}


def normalize_intent_level(level: str) -> str:
    key = str(level or "").strip().lower()
    return LEGACY_INTENT_LEVEL_ALIASES.get(key, key)


def normalize_prompt_id(prompt_id: str) -> str:
    key = str(prompt_id or "").strip().replace("\\", "/")
    return PROMPT_ID_ALIASES.get(key, key)
