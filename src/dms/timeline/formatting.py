from __future__ import annotations

from typing import Any


def format_timeline_report(graph: dict[str, Any]) -> str:
    lines = [
        "# Diegetic Timeline Report",
        "",
        "## Policy",
        f"- Version: {graph.get('version')}",
        "- Discourse order is a weak prior, not a world-time fact.",
        "- Reveal time remains separate from story time for benchmark anti-leak checks.",
        f"- Benchmark memory filtering changed: {str(graph.get('policy', {}).get('benchmark_memory_filtering_changed')).lower()}",
        "",
        "## Counts",
    ]
    counts = graph.get("counts") if isinstance(graph.get("counts"), dict) else {}
    for key in (
        "scene_count",
        "event_count",
        "main_story_event_count",
        "context_event_count",
        "forecast_event_count",
        "llm_relation_count",
        "ordering_relation_count",
        "selected_ordering_relation_count",
        "skipped_ordering_relation_count",
        "discourse_prior_relation_count",
        "conflict_count",
        "timeline_bucket_count",
    ):
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.extend(["", "## Timeline Buckets"])
    buckets = graph.get("timeline_buckets") if isinstance(graph.get("timeline_buckets"), list) else []
    if not buckets:
        lines.append("- No timeline buckets.")
    for bucket in buckets:
        scene_ids = ", ".join(bucket.get("scene_ids") or [])
        summaries = "; ".join(str(item) for item in (bucket.get("summaries") or []) if str(item).strip())
        lines.append(f"- {bucket.get('timeline_bucket')}: {scene_ids} | {summaries}")
    lines.extend(["", "## Non-Plot Tracks"])
    for label, key in (("Context", "context_timeline_order"), ("Forecast", "forecast_timeline_order")):
        items = graph.get(key) if isinstance(graph.get(key), list) else []
        lines.append(f"{label}: {len(items)}")
        for item in items[:12]:
            lines.append(f"- {item.get('event_id')}: {item.get('summary')}")
    lines.extend(["", "## Reveal-Time Risks"])
    reveal_risks = _reveal_time_risks(graph)
    if not reveal_risks:
        lines.append("- No world-time earlier / reveal-time later cases detected by the MVP graph.")
    for item in reveal_risks:
        lines.append(
            f"- {item.get('event_id')} revealed at {item.get('revealed_at_scene_id')}: {item.get('summary')}"
        )
    lines.extend(["", "## Skipped Ordering Edges"])
    skipped_edges = _skipped_ordering_edges(graph)
    if not skipped_edges:
        lines.append("- No usable ordering edges were skipped.")
    for relation in skipped_edges[:12]:
        lines.append(
            "- "
            f"{relation.get('relation_id')}: {relation.get('source_event_id')} -> {relation.get('target_event_id')} "
            f"({relation.get('source')}, {relation.get('ordering_skip_reason')})"
        )
    if len(skipped_edges) > 12:
        lines.append(f"- ... {len(skipped_edges) - 12} additional skipped edges omitted.")
    lines.extend(["", "## Conflicts"])
    conflicts = graph.get("conflicts") if isinstance(graph.get("conflicts"), list) else []
    if not conflicts:
        lines.append("- No hard ordering conflicts detected.")
    for conflict in conflicts:
        lines.append(f"- {conflict.get('type')}: {', '.join(str(item) for item in conflict.get('event_ids', []))}")
    lines.append("")
    return "\n".join(lines)


def _reveal_time_risks(graph: dict[str, Any]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    events = graph.get("events") if isinstance(graph.get("events"), list) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("is_reveal_of_past") or event.get("event_time_mode") == "past_recalled":
            risks.append(event)
    return risks


def _skipped_ordering_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
    return [
        relation
        for relation in relations
        if isinstance(relation, dict)
        and relation.get("usable_for_ordering")
        and relation.get("relation_type") in {"before", "after", "contains", "causes", "reveals_past"}
        and relation.get("selected_for_ordering") is False
    ]
