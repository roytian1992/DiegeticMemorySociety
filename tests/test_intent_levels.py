from dms.intent_levels import CLI_INTENT_LEVEL_CHOICES, normalize_intent_level, normalize_prompt_id


def test_normalize_intent_level_keeps_canonical_names() -> None:
    assert normalize_intent_level("social_simulation_intent") == "social_simulation_intent"
    assert normalize_intent_level("writing_intent") == "writing_intent"
    assert normalize_intent_level("writing_spec") == "writing_spec"


def test_normalize_intent_level_accepts_legacy_aliases() -> None:
    assert normalize_intent_level("sparse") == "social_simulation_intent"
    assert normalize_intent_level("detailed") == "writing_spec"
    assert normalize_intent_level("reference_scene_spec") == "writing_spec"


def test_normalize_prompt_id_accepts_reference_scene_spec_alias() -> None:
    assert normalize_prompt_id("dms/reference_scene_spec") == "dms/writing_spec"
    assert normalize_prompt_id(r"dms\reference_scene_spec.yaml") == "dms/writing_spec.yaml"


def test_cli_intent_level_choices_include_canonical_and_legacy_names() -> None:
    assert "social_simulation_intent" in CLI_INTENT_LEVEL_CHOICES
    assert "writing_intent" in CLI_INTENT_LEVEL_CHOICES
    assert "writing_spec" in CLI_INTENT_LEVEL_CHOICES
    assert "reference_scene_spec" in CLI_INTENT_LEVEL_CHOICES
    assert "sparse" in CLI_INTENT_LEVEL_CHOICES
    assert "detailed" in CLI_INTENT_LEVEL_CHOICES
