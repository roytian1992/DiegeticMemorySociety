import json
from pathlib import Path

from dms.author_context import format_author_entity_context_for_prompt, load_author_entity_context


def test_load_author_entity_context_normalizes_profiles(tmp_path: Path) -> None:
    path = tmp_path / "author_entities.json"
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "canonical_name": "张鹏",
                        "entity_type": "character",
                        "aliases": ["老张"],
                        "description": "作者设定里的战区老飞行员",
                        "author_profile": {
                            "stable_traits": ["嘴硬", "护短"],
                            "speaking_style": ["口语化", "带训诫感"],
                            "behavior_constraints": ["不能提前知道未来剧情"],
                        },
                        "initial_state": {
                            "beliefs": ["刘培强需要扛住返航压力"],
                            "relationships": {"刘培强": "保护和提醒"},
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    context = load_author_entity_context(path)
    entity = context["entities"][0]
    prompt_context = format_author_entity_context_for_prompt(context)

    assert entity["canonical_name"] == "张鹏"
    assert entity["author_description"] == "作者设定里的战区老飞行员"
    assert entity["author_profile"]["stable_traits"] == ["嘴硬", "护短"]
    assert entity["author_profile"]["speaking_style"] == ["口语化", "带训诫感"]
    assert entity["initial_state"]["relationships"]["刘培强"] == "保护和提醒"
    assert entity["profile_policy"]["priority"] == "author_locked"
    assert str(path) in entity["profile_sources"][1]["path"]
    assert "author_defined_descriptions_and_profiles_are_baselines" in prompt_context
    assert "不能提前知道未来剧情" in prompt_context
