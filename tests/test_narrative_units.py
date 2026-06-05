from dms.narrative_units import narrative_unit_identity, normalize_unit_label, normalize_unit_type


def test_narrative_unit_identity_defaults_to_scene_compatibility() -> None:
    identity = narrative_unit_identity(scene_id="scene_0007", discourse_index=7)

    assert identity["unit_id"] == "scene_0007"
    assert identity["unit_type"] == "scene"
    assert identity["unit_label"] == "scene"
    assert identity["unit_order"] == 7
    assert identity["parent_unit_id"] == "scene_0007"


def test_narrative_unit_type_accepts_author_chosen_labels() -> None:
    assert normalize_unit_type("Chapter Section") == "chapter_section"
    assert normalize_unit_label("", unit_type="chapter") == "chapter"

    identity = narrative_unit_identity(
        scene_id="scene_0002",
        discourse_index=2,
        unit_type="Chapter",
        unit_label="chapter",
    )

    assert identity["unit_id"] == "scene_0002"
    assert identity["unit_type"] == "chapter"
    assert identity["unit_label"] == "chapter"
    assert identity["parent_unit_type"] == "chapter"
