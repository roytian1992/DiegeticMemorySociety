from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime
from typing import Any

from dms.timeline.schema import TimelineBuildConfig

ORDERING_RELATIONS = {"before", "after", "contains", "causes", "reveals_past"}
SYMMETRIC_RELATIONS = {"overlaps", "same_time"}
MAIN_STORY_TRACKS = {"plot", "memory", "unknown"}


def build_timeline_graph(
    scene_outputs: list[dict[str, Any]],
    *,
    config: TimelineBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TimelineBuildConfig()
    events = [event for output in scene_outputs for event in output.get("temporal_events", [])]
    llm_relations = [relation for output in scene_outputs for relation in output.get("temporal_relations", [])]
    scene_temporal_index = [output.get("scene_temporal_index") for output in scene_outputs if output.get("scene_temporal_index")]
    event_by_id = {str(event.get("event_id")): event for event in events if event.get("event_id")}
    relations = _normalize_ordering_relations(llm_relations, event_by_id)
    if cfg.use_discourse_order_prior:
        relations.extend(_scene_block_prior_edges(events, cfg.scene_block_prior_confidence))
        relations.extend(_discourse_prior_edges(events, cfg.discourse_prior_confidence))
    conflicts = _detect_conflicts(relations, event_by_id, cfg)
    _mark_selected_ordering_relations(relations, conflicts, event_by_id)
    order = _timeline_order(events, relations, conflicts)
    buckets = _timeline_buckets(order, relations)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "diegetic_timeline_v0",
        "policy": {
            **asdict(cfg),
            "discourse_order_is_weak_prior": cfg.use_discourse_order_prior,
            "reveal_time_is_kept_separate": True,
            "benchmark_memory_filtering_changed": False,
        },
        "counts": {
            "scene_count": len(scene_outputs),
            "event_count": len(events),
            "main_story_event_count": len([event for event in events if _is_main_story_event(event)]),
            "context_event_count": len([event for event in events if event.get("event_track") == "context"]),
            "forecast_event_count": len([event for event in events if event.get("event_track") == "forecast"]),
            "llm_relation_count": len(llm_relations),
            "ordering_relation_count": len([item for item in relations if item.get("relation_type") in ORDERING_RELATIONS]),
            "selected_ordering_relation_count": len([item for item in relations if item.get("selected_for_ordering") is True]),
            "skipped_ordering_relation_count": len(
                [
                    item
                    for item in relations
                    if item.get("usable_for_ordering")
                    and item.get("relation_type") in ORDERING_RELATIONS
                    and item.get("selected_for_ordering") is False
                ]
            ),
            "discourse_prior_relation_count": len([item for item in relations if item.get("source") == "discourse_prior"]),
            "conflict_count": len(conflicts),
            "timeline_bucket_count": len(buckets),
        },
        "events": events,
        "relations": relations,
        "scene_temporal_index": scene_temporal_index,
        "timeline_order": order,
        "main_timeline_order": [item for item in order if item.get("event_track") in MAIN_STORY_TRACKS],
        "context_timeline_order": [item for item in order if item.get("event_track") == "context"],
        "forecast_timeline_order": [item for item in order if item.get("event_track") == "forecast"],
        "timeline_buckets": buckets,
        "conflicts": conflicts,
    }


def _normalize_ordering_relations(relations: list[dict[str, Any]], event_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for relation in relations:
        relation_type = str(relation.get("relation_type") or "uncertain")
        source = str(relation.get("source_event_id") or "")
        target = str(relation.get("target_event_id") or "")
        if not source or not target:
            normalized.append({**relation, "usable_for_ordering": False, "ordering_error": "missing endpoint"})
            continue
        if source not in event_by_id or target not in event_by_id:
            normalized.append({**relation, "usable_for_ordering": False, "ordering_error": "unknown endpoint"})
            continue
        if relation.get("audit_ordering_usable") is False:
            normalized.append({**relation, "usable_for_ordering": False, "ordering_error": "audit_unusable_evidence"})
            continue
        if relation_type == "after":
            normalized.append(
                {
                    **relation,
                    "source_event_id": target,
                    "target_event_id": source,
                    "relation_type": "before",
                    "original_relation_type": "after",
                    "usable_for_ordering": True,
                }
            )
            continue
        normalized.append(
            {
                **relation,
                "usable_for_ordering": relation_type in ORDERING_RELATIONS or relation_type in SYMMETRIC_RELATIONS,
            }
        )
    return normalized


def _scene_block_prior_edges(events: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in sorted(events, key=_event_sort_key):
        if not _is_main_story_event(event):
            continue
        scene_id = str(event.get("scene_id") or "")
        if scene_id:
            by_scene[scene_id].append(event)
    for scene_id, scene_events in by_scene.items():
        for left, right in zip(scene_events, scene_events[1:]):
            edges.append(
                {
                    "relation_id": f"scene_block_prior:{left.get('event_id')}->{right.get('event_id')}",
                    "scene_id": scene_id,
                    "source_record_id": left.get("source_record_id"),
                    "discourse_index": left.get("discourse_index"),
                    "source_event_id": left.get("event_id"),
                    "target_event_id": right.get("event_id"),
                    "relation_type": "before",
                    "evidence": "same scene event order prior",
                    "confidence": confidence,
                    "is_inferred": True,
                    "source": "scene_block_prior",
                    "usable_for_ordering": True,
                }
            )
    return edges


def _discourse_prior_edges(events: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    representative_by_scene: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=_event_sort_key):
        if not _is_main_story_event(event):
            continue
        scene_id = str(event.get("scene_id") or "")
        if scene_id and scene_id not in representative_by_scene:
            representative_by_scene[scene_id] = event
    ordered = sorted(representative_by_scene.values(), key=_event_sort_key)
    for index, left in enumerate(ordered[:-1], start=1):
        right = ordered[index]
        edges.append(
            {
                "relation_id": f"discourse_prior:{left.get('scene_id')}->{right.get('scene_id')}",
                "scene_id": left.get("scene_id"),
                "source_record_id": left.get("source_record_id"),
                "discourse_index": left.get("discourse_index"),
                "source_event_id": left.get("event_id"),
                "target_event_id": right.get("event_id"),
                "relation_type": "before",
                "evidence": "script chapter order prior",
                "confidence": confidence,
                "is_inferred": True,
                "source": "discourse_prior",
                "usable_for_ordering": True,
            }
        )
    return edges


def _detect_conflicts(
    relations: list[dict[str, Any]],
    event_by_id: dict[str, dict[str, Any]],
    config: TimelineBuildConfig,
) -> list[dict[str, Any]]:
    hard_edges = [
        relation
        for relation in relations
        if relation.get("usable_for_ordering")
        and relation.get("relation_type") in ORDERING_RELATIONS
        and float(relation.get("confidence") or 0) >= config.hard_relation_confidence
        and relation.get("source_event_id") in event_by_id
        and relation.get("target_event_id") in event_by_id
    ]
    conflicts: list[dict[str, Any]] = []
    graph: dict[str, set[str]] = defaultdict(set)
    edge_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in hard_edges:
        source = str(edge.get("source_event_id"))
        target = str(edge.get("target_event_id"))
        graph[source].add(target)
        edge_by_pair[(source, target)].append(edge)
        if source == target:
            conflicts.append(
                {
                    "type": "self_ordering_edge",
                    "event_id": source,
                    "relation_ids": [edge.get("relation_id")],
                }
            )
    for source, targets in graph.items():
        for target in targets:
            if source in graph.get(target, set()):
                conflicts.append(
                    {
                        "type": "two_node_cycle",
                        "event_ids": [source, target],
                        "relation_ids": [
                            *(edge.get("relation_id") for edge in edge_by_pair[(source, target)]),
                            *(edge.get("relation_id") for edge in edge_by_pair[(target, source)]),
                        ],
                    }
                )
    conflicts.extend(_cycle_conflicts(graph))
    return _dedupe_conflicts(conflicts)


def _mark_selected_ordering_relations(
    relations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    event_by_id: dict[str, dict[str, Any]],
) -> None:
    """Select an acyclic subset of usable ordering edges.

    LLM edges and discourse/scene priors can disagree. Priors are useful only
    as weak fallback edges, so timeline ordering should not let a low-confidence
    prior create a cycle that cancels stronger source-grounded evidence.
    """

    blocked_relation_ids = {
        str(relation_id)
        for conflict in conflicts
        for relation_id in (conflict.get("relation_ids") or [])
        if relation_id
    }
    candidates: list[dict[str, Any]] = []
    for relation in relations:
        relation_type = str(relation.get("relation_type") or "")
        relation["selected_for_ordering"] = False
        if not relation.get("usable_for_ordering") or relation_type not in ORDERING_RELATIONS:
            continue
        relation_id = str(relation.get("relation_id") or "")
        if relation_id in blocked_relation_ids:
            relation["ordering_skip_reason"] = "hard_conflict"
            continue
        source = str(relation.get("source_event_id") or "")
        target = str(relation.get("target_event_id") or "")
        if source not in event_by_id or target not in event_by_id:
            relation["ordering_skip_reason"] = "unknown_endpoint"
            continue
        if source == target:
            relation["ordering_skip_reason"] = "self_edge"
            continue
        candidates.append(relation)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in sorted(candidates, key=_ordering_relation_priority):
        source = str(relation.get("source_event_id") or "")
        target = str(relation.get("target_event_id") or "")
        if _path_exists(adjacency, target, source):
            relation["ordering_skip_reason"] = "would_create_cycle"
            continue
        adjacency[source].add(target)
        relation["selected_for_ordering"] = True
        relation.pop("ordering_skip_reason", None)


def _timeline_order(events: list[dict[str, Any]], relations: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_by_id = {str(event.get("event_id")): event for event in events if event.get("event_id")}
    blocked_relation_ids = {
        str(relation_id)
        for conflict in conflicts
        for relation_id in (conflict.get("relation_ids") or [])
        if relation_id
    }
    graph: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {event_id: 0 for event_id in event_by_id}
    support: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        if relation_id in blocked_relation_ids:
            continue
        if (
            not relation.get("usable_for_ordering")
            or relation.get("selected_for_ordering") is not True
            or relation.get("relation_type") not in ORDERING_RELATIONS
        ):
            continue
        source = str(relation.get("source_event_id") or "")
        target = str(relation.get("target_event_id") or "")
        if source not in event_by_id or target not in event_by_id or source == target:
            continue
        if target not in graph[source]:
            graph[source].add(target)
            indegree[target] = indegree.get(target, 0) + 1
            support[target].append(relation_id)
    ready = deque(sorted([event_id for event_id, degree in indegree.items() if degree == 0], key=lambda item: _event_sort_key(event_by_id[item])))
    ordered_ids: list[str] = []
    while ready:
        event_id = ready.popleft()
        ordered_ids.append(event_id)
        for target in sorted(graph.get(event_id, set()), key=lambda item: _event_sort_key(event_by_id[item])):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    for event_id in sorted(event_by_id, key=lambda item: _event_sort_key(event_by_id[item])):
        if event_id not in ordered_ids:
            ordered_ids.append(event_id)
    return [
        {
            "timeline_rank": index,
            "event_id": event_id,
            "scene_id": event_by_id[event_id].get("scene_id"),
            "source_record_id": event_by_id[event_id].get("source_record_id"),
            "summary": event_by_id[event_id].get("summary"),
            "event_track": event_by_id[event_id].get("event_track"),
            "story_time_hint": event_by_id[event_id].get("story_time_hint"),
            "event_time_mode": event_by_id[event_id].get("event_time_mode"),
            "revealed_at_scene_id": event_by_id[event_id].get("revealed_at_scene_id"),
            "ordering_support": support.get(event_id, []),
        }
        for index, event_id in enumerate(ordered_ids, start=1)
    ]


def _timeline_buckets(order: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    same_or_overlap: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.get("relation_type") not in SYMMETRIC_RELATIONS or not relation.get("usable_for_ordering"):
            continue
        source = str(relation.get("source_event_id") or "")
        target = str(relation.get("target_event_id") or "")
        if not source or not target:
            continue
        same_or_overlap[source].add(target)
        same_or_overlap[target].add(source)
    by_event = {item["event_id"]: item for item in order}
    visited: set[str] = set()
    buckets: list[dict[str, Any]] = []
    for item in order:
        event_id = item["event_id"]
        if event_id in visited:
            continue
        group = sorted(_connected_component(event_id, same_or_overlap), key=lambda current: by_event.get(current, {}).get("timeline_rank", 10**9))
        if not group:
            group = [event_id]
        visited.update(group)
        bucket_events = [by_event[current] for current in group if current in by_event]
        buckets.append(
            {
                "timeline_bucket": f"T{len(buckets) + 1:03d}",
                "event_ids": group,
                "scene_ids": sorted({str(event.get("scene_id")) for event in bucket_events if event.get("scene_id")}),
                "summaries": [event.get("summary") for event in bucket_events],
            }
        )
    return buckets


def _connected_component(start: str, graph: dict[str, set[str]]) -> set[str]:
    stack = [start]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(sorted(graph.get(current, set()) - visited))
    return visited


def _cycle_conflicts(graph: dict[str, set[str]]) -> list[dict[str, Any]]:
    visited: set[str] = set()
    stack: set[str] = set()
    conflicts: list[dict[str, Any]] = []

    def visit(node: str, path: list[str]) -> None:
        if node in stack:
            cycle_start = path.index(node) if node in path else 0
            conflicts.append({"type": "cycle", "event_ids": path[cycle_start:] + [node], "relation_ids": []})
            return
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        for child in graph.get(node, set()):
            visit(child, [*path, child])
        stack.remove(node)

    for node in sorted(graph):
        visit(node, [node])
    return conflicts


def _dedupe_conflicts(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for conflict in conflicts:
        key = (
            conflict.get("type"),
            tuple(sorted(str(item) for item in conflict.get("event_ids", []))),
            tuple(sorted(str(item) for item in conflict.get("relation_ids", []))),
            conflict.get("event_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(conflict)
    return deduped


def _ordering_relation_priority(relation: dict[str, Any]) -> tuple[float, int, int, str]:
    try:
        confidence = float(relation.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    source_priority = {
        "llm": 3,
        "scene_block_prior": 2,
        "discourse_prior": 1,
    }.get(str(relation.get("source") or ""), 0)
    observed_priority = 0 if relation.get("is_inferred") else 1
    return (-confidence, -source_priority, -observed_priority, str(relation.get("relation_id") or ""))


def _path_exists(graph: dict[str, set[str]], source: str, target: str) -> bool:
    if source == target:
        return True
    stack = [source]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(sorted(graph.get(current, set()) - visited))
    return False


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int, str]:
    source_record_id = event.get("source_record_id")
    discourse_index = event.get("discourse_index")
    return (
        int(source_record_id) if isinstance(source_record_id, int) else 10**9,
        int(discourse_index) if isinstance(discourse_index, int) else 10**9,
        str(event.get("event_id") or ""),
    )


def _is_main_story_event(event: dict[str, Any]) -> bool:
    return str(event.get("event_track") or "unknown") in MAIN_STORY_TRACKS
