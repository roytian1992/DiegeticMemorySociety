from dms.prompts import YAMLPromptLoader


def test_yaml_prompt_loader_renders_nested_prompt_id() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/scene_inventory",
        task_values={"unit_json": {"unit_id": "scene_0001", "content": "测试文本"}},
        static_values={"extraction_policy": "Use only the input text."},
    )

    assert "Fill the inventory JSON" in rendered
    assert "scene_0001" in rendered
    assert "Use only the input text" in rendered


def test_yaml_prompt_loader_rejects_missing_required_values() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")

    try:
        loader.render("dms/scene_inventory", task_values={}, static_values={})
    except ValueError as exc:
        assert "unit_json" in str(exc)
        assert "extraction_policy" in str(exc)
    else:
        raise AssertionError("Expected missing required values to fail")


def test_yaml_prompt_loader_renders_kg_entity_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/kg_entity_mentions",
        task_values={"unit_json": {"unit_id": "scene_0001", "content": "刘培强进入机房。"}},
        static_values={"extraction_policy": "Use only the input text.", "entity_type_policy": "character: a person."},
    )

    assert "Fill the entity mention JSON" in rendered
    assert "entity_mentions" in rendered
    assert '"description": "string"' in rendered
    assert "description is the entity's reusable identity" in rendered
    assert "role_in_unit is its local narrative function" in rendered
    assert "attributes_or_state is the state or attribute stated in this unit" in rendered
    assert "scene_0001" in rendered


def test_yaml_prompt_loader_renders_scene_summary_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/scene_summary",
        task_values={"unit_json": {"unit_id": "scene_0001", "content": "刘培强进入机房。"}},
        static_values={"summary_policy": "Write a compact recap."},
    )

    assert "Write a concise summary record" in rendered
    assert '"retrieval_text": "string"' in rendered
    assert "Do not add fields such as seed_entities" in rendered
    assert "Write a compact recap" in rendered
    assert "scene_0001" in rendered


def test_yaml_prompt_loader_renders_reference_items_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/reference_items",
        task_values={
            "reference_chunk_json": {
                "chunk_id": "ref_doc_0001_chunk_0001",
                "title": "设定笔记",
                "content": "550A位于数字生命研究室。",
            }
        },
        static_values={"extraction_policy": "Every item needs evidence."},
    )

    assert "Extract reusable external reference items" in rendered
    assert '"reference_items": [' in rendered
    assert "knowledge_scope=author_only" in rendered
    assert "Every item needs evidence" in rendered
    assert "ref_doc_0001_chunk_0001" in rendered


def test_yaml_prompt_loader_renders_kg_entity_refinement_prompt_without_full_text() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/kg_entity_refinement",
        task_values={
            "unit_context_json": {"unit_id": "scene_0001", "title": "测试"},
            "scene_inventory_json": {"stated_facts": []},
            "initial_entity_json": {"entity_mentions": []},
        },
        static_values={"refinement_policy": "Classify only.", "entity_type_policy": "character: a person."},
    )

    assert "Refine the initial entity extraction" in rendered
    assert "full source text is intentionally not provided" in rendered
    assert '"description": "string"' in rendered
    assert "Keep it separate from the entity's local role and current-unit state" in rendered
    assert "scene_0001" in rendered


def test_yaml_prompt_loader_renders_writing_intent_prompt_with_author_anchors_but_no_extra_fields() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/writing_intent",
        task_values={"unit_json": {"unit_id": "scene_0006", "content": "返程场景。"}},
        static_values={"intent_policy": "Output only writing_intent."},
    )

    assert "Extract a realistic author writing intent" in rendered
    assert '"writing_intent": "string"' in rendered
    assert "Keep concrete anchors that an author would naturally know" in rendered
    assert "Return no fields beyond writing_intent" in rendered
    assert "Do not add unit_id" in rendered
    assert "Do not include retrieval instructions" in rendered
    assert "Output only writing_intent" in rendered
    assert "scene_0006" in rendered


def test_yaml_prompt_loader_renders_writing_spec_prompt_for_evaluation() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/writing_spec",
        task_values={"unit_json": {"unit_id": "scene_0006", "content": "返程场景。"}},
        static_values={"intent_policy": "Keep compact evaluation requirements."},
    )

    assert "Extract a concise writing specification" in rendered
    assert "benchmark ground truth for evaluation only" in rendered
    assert '"writing_spec"' in rendered
    assert '"required_narrative_units": ["string"]' in rendered
    assert "must never exceed the source passage length" in rendered
    assert "captures the core scene function" in rendered
    assert "Do not require broad geography, destination organizations, route labels" in rendered
    assert "prefer \"dangerous or unstable flying\" over an exact bridge" in rendered
    assert "preserve the behavior function and character pressure before the physical mechanics" in rendered
    assert "Do not add unit_id" in rendered


def test_yaml_prompt_loader_accepts_reference_scene_spec_alias() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/reference_scene_spec",
        task_values={"unit_json": {"unit_id": "scene_0006", "content": "返程场景。"}},
        static_values={"intent_policy": "Keep compact evaluation requirements."},
    )

    assert '"writing_spec"' in rendered
    assert "Do not include exact dialogue" in rendered
    assert "Keep compact evaluation requirements" in rendered
    assert "scene_0006" in rendered


def test_yaml_prompt_loader_renders_social_simulation_intent_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/social_simulation_intent",
        task_values={"unit_json": {"unit_id": "scene_0006", "content": "返程场景。"}},
        static_values={"intent_policy": "Keep only low-information setup."},
    )

    assert '"social_simulation_intent": "string"' in rendered
    assert "Extract a low-information social simulation intent" in rendered
    assert "less informative than the normal writing intent" in rendered
    assert "Return no fields beyond social_simulation_intent" in rendered
    assert "Keep only low-information setup" in rendered
    assert "scene_0006" in rendered


def test_yaml_prompt_loader_renders_generic_writing_generation_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/writing_generation",
        task_values={
            "writing_request": "写一段返航途中人物互动。",
            "memory_packet": "# Memory Packet\n\n## Entities\n...",
            "previous_scene_context": "Previous scene: scene_0005\nFull text:\n上一场景内容。",
            "style_reference": "张鹏：慢点慢点。",
            "length_requirement": "正文必须为127-182个中文字符。",
            "output_requirements": "- 中文输出\n- 必须包含UEG",
        },
    )

    assert "Write the next narrative passage according to the writing request." in rendered
    assert "写一段返航途中人物互动。" in rendered
    assert rendered.count("# Memory Packet") == 1
    assert "# Previous Scene Context" in rendered
    assert "上一场景内容。" in rendered
    assert "张鹏：慢点慢点。" in rendered
    assert "# Length Requirement" in rendered
    assert "正文必须为127-182个中文字符。" in rendered
    assert "Treat previous scene context as an auxiliary continuity reference only" in rendered
    assert "never let it replace, weaken, or omit anchors from the writing request" in rendered
    assert "Preserve explicit named anchors from the writing request" in rendered
    assert "Use the style reference only for prose format" in rendered
    assert "必须包含UEG" in rendered
    assert "Show 刘培强" not in rendered
    assert "J20C" not in rendered


def test_yaml_prompt_loader_renders_social_writing_generation_prompt_separately() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/writing_generation_social",
        task_values={
            "writing_request": "写一段返航途中人物互动。",
            "memory_packet": "# Memory Packet\n\n## Entities\n...",
            "attribute_cards": "# Entity Attribute Cards\n\n## 刘培强",
            "social_simulation": (
                "# Social Simulation Writer Packet\n\n"
                "## Optional Interaction Functions\n"
                "- b_001: value_resistance; participants: 刘培强, 张鹏\n"
                "  action: 刘培强 / risky_operation / 压低飞行姿态\n"
                "  note: posture only; do not copy as final dialogue\n"
            ),
            "previous_scene_context": "Previous scene: scene_0005\nSummary:\n上一场景摘要。",
            "style_reference": "张鹏：慢点慢点。",
            "length_requirement": "正文必须为127-182个中文字符。",
            "output_requirements": "- 中文输出\n- 必须包含UEG",
        },
    )

    assert "Write the next narrative passage according to the writing request." in rendered
    assert "# Attribute Cards" in rendered
    assert "# Social Simulation" in rendered
    assert "# Previous Scene Context" in rendered
    assert "上一场景摘要。" in rendered
    assert "Writing request is the primary creative target" in rendered
    assert "must never replace, weaken, or omit anchors from the writing request" in rendered
    assert "Preserve explicit named anchors from the writing request" in rendered
    assert "Social simulation is a writing aid, not a fact source" in rendered
    assert "Optional Interaction Functions as optional scene functions" in rendered
    assert "Dialogue Posture as communicative intent and tone only" in rendered
    assert "Avoid phrases in the social simulation are prohibitions" in rendered
    assert "Do not copy writer-packet wording, beat ids, action labels" in rendered
    assert "Never treat posture text as canonical dialogue" in rendered
    assert "posture only; do not copy as final dialogue" in rendered
    assert "old-soldier voice" in rendered
    assert "do not write Zhang Peng as a legal guardian" in rendered
    assert rendered.count("# Memory Packet") == 1
    assert "正文必须为127-182个中文字符。" in rendered


def test_yaml_prompt_loader_renders_writing_evaluation_prompts() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    requirements = loader.render(
        "dms/eval_intent_requirements",
        task_values={"writing_intent": "写一段刘培强和张鹏返航。"},
    )
    consistency = loader.render(
        "dms/eval_intent_consistency",
        task_values={
            "writing_intent": "写一段刘培强和张鹏返航。",
            "requirements_json": {"requirements": []},
            "candidate_label": "generated",
            "candidate_text": "刘培强驾驶J20C。",
        },
    )
    quality = loader.render(
        "dms/eval_writing_quality",
        task_values={
            "writing_intent": "写一段刘培强和张鹏返航。",
            "candidate_label": "generated",
            "candidate_text": "刘培强驾驶J20C。",
        },
    )
    faithfulness = loader.render(
        "dms/eval_memory_faithfulness",
        task_values={
            "writing_intent": "写一段刘培强和张鹏返航。",
            "memory_packet": "# Memory Packet",
            "candidate_label": "generated",
            "candidate_text": "刘培强驾驶J20C。",
        },
    )

    assert "Decompose the writing intent" in requirements
    assert "treat \"Required ...\" fields as the authoritative checklist" in requirements
    assert "use \"Scene purpose\" only as context" in requirements
    assert "FactScore-like checklist" in consistency
    assert "Evaluate the writing quality" in quality
    assert "Evaluate memory faithfulness" in faithfulness


def test_yaml_prompt_loader_renders_entity_attribute_card_prompt() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    rendered = loader.render(
        "dms/entity_attribute_card",
        task_values={
            "entity_context_json": {
                "entity": {"canonical_name": "刘培强", "entity_type": "character"},
                "related_memories": [{"index": "M1", "summary": "测试记忆"}],
            }
        },
    )

    assert "Build a writing-facing attribute card" in rendered
    assert "Distinguish explicit facts from inferred traits" in rendered
    assert '"salient_past_actions"' in rendered
    assert '"stable_traits"' in rendered
    assert '"hard_constraints"' in rendered
    assert '"simulation_risks"' in rendered
    assert "刘培强" in rendered


def test_yaml_prompt_loader_renders_social_simulation_prompts() -> None:
    loader = YAMLPromptLoader("task_specs/prompts")
    disposition_prompt = loader.render(
        "dms/scene_disposition_note",
        task_values={
            "disposition_context_json": {
                "social_simulation_intent": "写一段返航互动。",
                "target_card": {"canonical_name": "刘培强"},
            }
        },
    )
    character_prompt = loader.render(
        "dms/character_social_simulation",
        task_values={
            "simulation_context_json": {
                "writing_intent": "写一段返航互动。",
                "target_card": {"canonical_name": "刘培强"},
                "relevant_memory_notes": ["M1: 刘培强返航途中情绪焦躁"],
            }
        },
    )
    coordinator_prompt = loader.render(
        "dms/social_simulation_coordinator",
        task_values={
            "coordinator_context_json": {
                "writing_intent": "写一段返航互动。",
                "character_simulations": [{"character": "刘培强"}],
            }
        },
    )

    assert "Write one compact scene disposition note" in disposition_prompt
    assert "Use relevant_memory_notes when present" in disposition_prompt
    assert "Return exactly these three fields" in disposition_prompt
    assert '"scene_disposition_note": "string"' in disposition_prompt
    assert "刘培强" in disposition_prompt
    assert "Simulate how the target character" in character_prompt
    assert '"likely_internal_state"' in character_prompt
    assert '"intent_assumptions"' in character_prompt
    assert "刘培强" in character_prompt
    assert "Coordinate the character simulations" in coordinator_prompt
    assert '"scene_beats"' in coordinator_prompt
    assert '"intent_basis"' in coordinator_prompt
