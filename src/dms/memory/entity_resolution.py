from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dms.entity_types import normalize_entity_type
from dms.memory.world_model import load_prefix_world_model
from dms.relationship_types import (
    canonicalize_relation_type,
    is_durable_relation_type,
    is_undirected_relation_type,
    soften_formal_relation_type,
)


_TITLE_PREFIXES = (
    "Dr.",
    "Dr",
    "Mr.",
    "Mr",
    "Mrs.",
    "Mrs",
    "Ms.",
    "Ms",
    "Captain",
    "Commander",
    "Professor",
    "Prof.",
    "Prof",
    "Officer",
    "队长",
    "博士",
    "教授",
    "老师",
    "先生",
    "女士",
)
_GROUP_OR_ROLE_NAMES = {"科研人员", "观众", "镜头观众", "所有场景内角色"}
_ROLE_SUFFIXES = ("人员", "观众", "设备")
_COMMON_CHINESE_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟"
    "平黄和穆萧尹姚邵汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝"
    "董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田"
    "胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左"
    "石崔吉龚程邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富"
    "巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖"
    "武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠"
    "蒙池乔阴郁胥苍闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿"
    "通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈"
    "廖庾终暨居衡步都耿满弘匡国文寇广禄阙东殴殳沃利蔚越夔隆师巩厍"
    "聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游"
    "竺权逯盖益桓公"
)
_COMPOUND_CHINESE_SURNAMES = {
    "欧阳",
    "太史",
    "端木",
    "上官",
    "司马",
    "东方",
    "独孤",
    "南宫",
    "万俟",
    "闻人",
    "夏侯",
    "诸葛",
    "尉迟",
    "公羊",
    "赫连",
    "澹台",
    "皇甫",
    "宗政",
    "濮阳",
    "公冶",
    "太叔",
    "申屠",
    "公孙",
    "慕容",
    "仲孙",
    "钟离",
    "长孙",
    "宇文",
    "司徒",
    "鲜于",
    "司空",
    "闾丘",
    "子车",
    "亓官",
    "司寇",
    "巫马",
    "公西",
    "颛孙",
    "壤驷",
    "公良",
    "漆雕",
    "乐正",
    "宰父",
    "谷梁",
    "拓跋",
    "夹谷",
    "轩辕",
    "令狐",
    "段干",
    "百里",
    "呼延",
    "东郭",
    "南门",
    "羊舌",
    "微生",
}
_NON_PERSON_NAME_TERMS = (
    "科学家",
    "受试者",
    "科研",
    "人员",
    "观众",
    "镜头",
    "设备",
    "数据",
    "信息",
    "计算机",
    "备份卡",
    "屏",
    "屏幕",
    "贴片",
    "接口",
    "解说",
    "旁白",
    "录制",
    "电波",
    "脑电波",
    "信息",
    "概念",
    "记忆",
    "实验",
    "活动",
    "会议",
    "计划",
    "行动",
)
_DEICTIC_TERMS = {
    "这",
    "这个",
    "这儿",
    "这里",
    "那里",
    "那儿",
    "它",
    "他",
    "她",
    "他们",
    "她们",
    "它们",
    "this",
    "that",
    "here",
    "there",
    "it",
    "he",
    "she",
    "they",
}
_DEICTIC_PREFIXES = ("这", "那", "this ", "that ")
_TYPE_PRIORITY = (
    "character",
    "group",
    "organization",
    "object",
    "location",
    "occasion",
    "concept",
)
def build_entity_resolution_artifacts(
    world_model_path_or_dir: str | Path,
    output_dir: str | Path,
    *,
    author_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an auditable entity registry and relation timeline from a prefix world model."""

    world_model = load_prefix_world_model(world_model_path_or_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mentions = [*_author_entity_mentions(author_entities or []), *_collect_mentions(world_model)]
    entities, alias_records, resolution_traces = _resolve_mentions(mentions)
    relationship_updates = _extract_relationship_updates(world_model, entities)
    relationships = _build_relationship_states(relationship_updates)

    paths = {
        "entities": out_path / "entities.jsonl",
        "aliases": out_path / "aliases.jsonl",
        "resolution_traces": out_path / "resolution_traces.jsonl",
        "relationship_updates": out_path / "relationship_updates.jsonl",
        "relationships": out_path / "relationships.jsonl",
        "summary": out_path / "summary.json",
    }
    _write_jsonl(paths["entities"], entities)
    _write_jsonl(paths["aliases"], alias_records)
    _write_jsonl(paths["resolution_traces"], resolution_traces)
    _write_jsonl(paths["relationship_updates"], relationship_updates)
    _write_jsonl(paths["relationships"], relationships)

    summary = {
        "source_world_model": str(Path(world_model_path_or_dir)),
        "output_dir": str(out_path),
        "entity_count": len(entities),
        "alias_count": len(alias_records),
        "resolution_trace_count": len(resolution_traces),
        "relationship_update_count": len(relationship_updates),
        "relationship_count": len(relationships),
        "artifact_paths": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def resolve_name_variants(name: str) -> list[str]:
    """Generate deterministic alias variants for Chinese and English names."""

    clean = _strip_title(str(name or "").strip())
    if not clean:
        return []
    variants = {clean, _normalize_space(clean)}

    no_parens = re.sub(r"[（(].*?[）)]", "", clean).strip()
    if no_parens:
        variants.add(no_parens)

    if _contains_cjk(clean):
        variants.update(_chinese_variants(clean))
    else:
        variants.update(_latin_variants(clean))

    return sorted({variant for variant in variants if variant})


def _collect_mentions(world_model: dict[str, Any]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    for record in _as_list(world_model.get("characters")):
        name = str(record.get("canonical_name") or "").strip()
        if name:
            mentions.append(_mention(name, "character", record.get("scene_ids", [None])[0], "canonical_character", record))
        for item in _as_list(record.get("mentions")):
            item_name = str(item.get("name") or "").strip()
            if item_name:
                mentions.append(_mention(item_name, "character", item.get("scene_id"), "character_mention", item))

    for record in _as_list(world_model.get("objects")):
        name = str(record.get("canonical_name") or "").strip()
        if name:
            mentions.append(_mention(name, "object", record.get("scene_ids", [None])[0], "canonical_object", record))
        for item in _as_list(record.get("mentions")):
            item_name = str(item.get("name") or "").strip()
            if item_name:
                mentions.append(_mention(item_name, "object", item.get("scene_id"), "object_mention", item))

    for record in _as_list(world_model.get("kg_entity_mentions")):
        surface = str(record.get("surface") or "").strip()
        if surface:
            mentions.append(
                _mention(
                    surface,
                    normalize_entity_type(record.get("entity_type") or _guess_entity_type(surface)),
                    record.get("scene_id"),
                    "kg_entity_mention",
                    record,
                    canonical_hint=str(record.get("canonical_hint") or ""),
                    description=str(record.get("description") or ""),
                )
            )

    for event in _as_list(world_model.get("events")):
        for participant in _as_list(event.get("participants")):
            name = str(participant or "").strip()
            if name:
                mentions.append(_mention(name, _guess_entity_type(name), event.get("scene_id"), "event_participant", event))

    for transfer in _as_list(world_model.get("knowledge_transfers")):
        for field in ("source", "receiver"):
            name = str(transfer.get(field) or "").strip()
            if name:
                mentions.append(_mention(name, _guess_entity_type(name), transfer.get("scene_id"), f"knowledge_transfer_{field}", transfer))

    for state_change in _as_list(world_model.get("state_changes")):
        name = str(state_change.get("entity") or "").strip()
        if name:
            mentions.append(_mention(name, _guess_entity_type(name), state_change.get("scene_id"), "state_change_entity", state_change))

    for link in _as_list(world_model.get("entity_memory_links")):
        name = str(link.get("entity") or "").strip()
        if name:
            mentions.append(
                _mention(
                    name,
                    normalize_entity_type(link.get("entity_type") or _guess_entity_type(name)),
                    link.get("scene_id"),
                    "entity_memory_link",
                    link,
                )
            )

    for visibility in _as_list(world_model.get("visibility_records")):
        name = str(visibility.get("character") or "").strip()
        if name:
            mentions.append(_mention(name, _guess_entity_type(name), visibility.get("scene_id"), "visibility_character", visibility))

    return mentions


def _resolve_mentions(mentions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        if mention.get("invalid_surface"):
            continue
        key_name = str(mention.get("canonical_hint") or mention["name"])
        canonical_key = _canonical_key_for_mention(key_name, mention)
        grouped[canonical_key].append(mention)

    entities: list[dict[str, Any]] = []
    alias_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for index, (key, items) in enumerate(sorted(grouped.items()), start=1):
        names = [str(item["name"]) for item in items]
        canonical_candidates = [str(item.get("canonical_hint")) for item in items if str(item.get("canonical_hint") or "").strip()]
        entity_type = _choose_entity_type([str(item["entity_type"]) for item in items])
        canonical_name = _choose_canonical_name(canonical_candidates or names)
        descriptions = _choose_descriptions(items)
        author_descriptions = _choose_descriptions([item for item in items if item.get("source") == "author_defined"])
        entity_id = f"{entity_type}_{index:04d}"
        alias_names = [*names, *canonical_candidates]
        aliases = sorted({alias for name in alias_names for alias in resolve_name_variants(name)})
        scene_ids = sorted({str(item["scene_id"]) for item in items if item.get("scene_id")})
        entity = {
            "memory_layer": "entity_registry",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "name_parts": _name_parts(canonical_name),
            "language": "zh" if _contains_cjk(canonical_name) else "latin",
            "first_seen_scene": scene_ids[0] if scene_ids else None,
            "scene_ids": scene_ids,
            "mention_count": len(items),
            "descriptions": descriptions,
            "author_description": author_descriptions[0] if author_descriptions else "",
            "initial_description": author_descriptions[0] if author_descriptions else (descriptions[0] if descriptions else ""),
            "description_sources": _description_sources(items),
        }
        entities.append(entity)
        for alias in aliases:
            alias_records.append(
                {
                    "entity_id": entity_id,
                    "alias": alias,
                    "normalized_alias": _normalize_name(alias),
                    "source": "rule_name_variant" if alias != canonical_name else "canonical",
                }
            )
        for item in items:
            traces.append(
                {
                    "entity_id": entity_id,
                    "mention": item["name"],
                    "canonical_hint": item.get("canonical_hint", ""),
                    "entity_type": entity_type,
                    "scene_id": item.get("scene_id"),
                    "source": item.get("source"),
                    "resolution_method": "normalized_alias_rule",
                    "confidence": 1.0 if _normalize_name(item["name"]) == _normalize_name(canonical_name) else 0.85,
                    "description": item.get("description", ""),
                    "evidence": item.get("evidence", ""),
                }
            )
    return entities, alias_records, traces


def _extract_relationship_updates(world_model: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alias_index = _alias_index(entities)
    updates: list[dict[str, Any]] = []

    for observation in _as_list(world_model.get("relationship_observations")):
        source = _resolve_entity(str(observation.get("source_entity", "")), alias_index)
        target = _resolve_entity(str(observation.get("target_entity", "")), alias_index)
        relation_type, reverse_endpoints = canonicalize_relation_type(
            observation.get("relation_type") or "relationship_observation"
        )
        relation_type = soften_formal_relation_type(
            relation_type,
            evidence=observation.get("evidence", ""),
            status_or_change=observation.get("status_or_change", ""),
        )
        if reverse_endpoints:
            source, target = target, source
        if source and target and is_durable_relation_type(relation_type):
            updates.append(
                _relationship_update(
                    source,
                    target,
                    relation_type,
                    observation.get("scene_id"),
                    observation.get("record_id"),
                    "relationship_observation",
                    observation.get("status_or_change", ""),
                    observation.get("evidence", ""),
                    strength_delta=0.25,
                    direction=_relationship_direction(relation_type, source, target),
                )
            )

    updates.sort(key=lambda item: (str(item.get("scene_id")), str(item.get("record_id"))))
    for index, update in enumerate(updates, start=1):
        update["update_id"] = f"rel_update_{index:04d}"
    return updates


def _build_relationship_states(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        key = (str(update["source_entity_id"]), str(update["target_entity_id"]), str(update["relation_type"]))
        if update.get("direction") == "undirected" and key[0] > key[1]:
            key = (key[1], key[0], key[2])
        grouped[key].append(update)

    relationships: list[dict[str, Any]] = []
    for index, ((source_id, target_id, relation_type), items) in enumerate(sorted(grouped.items()), start=1):
        strengths = [float(item.get("strength_delta", 0.0)) for item in items]
        scene_ids = sorted({str(item.get("scene_id")) for item in items if item.get("scene_id")})
        relationships.append(
            {
                "memory_layer": "relationship_timeline",
                "relationship_id": f"relationship_{index:04d}",
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "relation_type": relation_type,
                "direction": items[-1].get("direction", "undirected"),
                "status": "active",
                "first_seen_scene": scene_ids[0] if scene_ids else None,
                "last_updated_scene": scene_ids[-1] if scene_ids else None,
                "strength": round(sum(strengths), 3),
                "update_ids": [item["update_id"] for item in items],
                "evidence": [item.get("evidence", "") for item in items if item.get("evidence")],
            }
        )
    return relationships


def _relationship_update(
    source: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
    scene_id: Any,
    record_id: Any,
    source_record_type: str,
    summary: Any,
    evidence: Any,
    *,
    strength_delta: float,
    direction: str = "undirected",
    epistemic_status: Any = "",
) -> dict[str, Any]:
    return {
        "memory_layer": "relationship_update",
        "update_id": "",
        "source_entity_id": source["entity_id"],
        "source_name": source["canonical_name"],
        "target_entity_id": target["entity_id"],
        "target_name": target["canonical_name"],
        "relation_type": relation_type,
        "direction": direction,
        "scene_id": scene_id,
        "source_record_id": record_id,
        "source_record_type": source_record_type,
        "summary": summary,
        "epistemic_status": epistemic_status,
        "strength_delta": strength_delta,
        "evidence": evidence,
    }


def _relationship_direction(relation_type: str, source: dict[str, Any], target: dict[str, Any]) -> str:
    if source["entity_id"] == target["entity_id"]:
        return "self"
    return "undirected" if is_undirected_relation_type(relation_type) else "directed"


def _mention(
    name: str,
    entity_type: str,
    scene_id: Any,
    source: str,
    record: dict[str, Any],
    *,
    canonical_hint: str = "",
    description: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "entity_type": entity_type,
        "scene_id": scene_id,
        "source": source,
        "canonical_hint": canonical_hint,
        "description": description,
        "evidence": record.get("evidence", "") or record.get("summary", "") or record.get("content", ""),
        "invalid_surface": _is_invalid_entity_surface(name, record),
    }


def _author_entity_mentions(author_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for entity in author_entities:
        if not isinstance(entity, dict):
            continue
        canonical_name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
        if not canonical_name:
            continue
        entity_type = normalize_entity_type(entity.get("entity_type") or entity.get("type") or "character")
        description = str(entity.get("author_description") or entity.get("description") or "").strip()
        mentions.append(
            _mention(
                canonical_name,
                entity_type,
                None,
                "author_defined",
                {"evidence": "", "description": description},
                canonical_hint=canonical_name,
                description=description,
            )
        )
        for alias in _as_list(entity.get("aliases")):
            alias_text = str(alias or "").strip()
            if alias_text and alias_text != canonical_name:
                mentions.append(
                    _mention(
                        alias_text,
                        entity_type,
                        None,
                        "author_defined",
                        {"evidence": "", "description": description},
                        canonical_hint=canonical_name,
                        description=description,
                    )
                )
    return mentions


def _is_invalid_entity_surface(name: object, record: dict[str, Any]) -> bool:
    clean = str(name or "").strip()
    if not clean:
        return True
    if "/" in clean and clean not in str(record.get("evidence", "")):
        return True
    return False


def _guess_entity_type(name: str) -> str:
    if name in _GROUP_OR_ROLE_NAMES:
        return "group"
    if any(term in name for term in ("计算机", "接口", "系统", "设施", "平台", "实验室", "研究室")):
        return "object" if not any(term in name for term in ("实验室", "研究室")) else "location"
    if any(term in name for term in ("卡", "设备", "屏", "线", "贴片", "镜头")):
        return "object"
    if any(term in name for term in ("数据", "信息", "记忆", "感知", "电信号", "脑电波", "意识", "概念")):
        return "concept"
    if any(term in name for term in ("计划", "行动", "活动", "会议", "实验", "事故", "灾难", "事件")):
        return "occasion"
    return "character"


def _canonical_key(name: str) -> str:
    variants = resolve_name_variants(name)
    if not variants:
        return _normalize_name(name)
    return min(_normalize_name(variant) for variant in variants)


def _canonical_key_for_mention(name: str, mention: dict[str, Any]) -> str:
    base = _canonical_key(name)
    if _is_deictic_name(name):
        return f"deictic:{base}:{mention.get('scene_id')}:{mention.get('source')}:{_normalize_name(mention.get('evidence', ''))[:24]}"

    entity_type = normalize_entity_type(mention.get("entity_type"))
    if entity_type == "object":
        code_key = _embedded_alnum_code_key(name)
        if code_key:
            return f"code:{code_key}"

    if entity_type in {"character", "group"} and _contains_cjk(name):
        role_key = _role_entity_key(name)
        if role_key:
            return f"role:{role_key}"

    return base


def _choose_canonical_name(names: list[str]) -> str:
    non_deictic = [name for name in names if not _is_deictic_name(name)]
    candidates = non_deictic or names
    return sorted(candidates, key=lambda value: (-len(value), value))[0]


def _choose_descriptions(items: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    descriptions: list[str] = []
    seen: set[str] = set()
    for item in sorted(items, key=lambda value: str(value.get("scene_id") or "")):
        description = re.sub(r"\s+", " ", str(item.get("description") or "").strip())
        if not description:
            continue
        key = _normalize_name(description)
        if key in seen:
            continue
        descriptions.append(description)
        seen.add(key)
        if len(descriptions) >= limit:
            break
    return descriptions


def _description_sources(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(items, key=lambda value: (0 if value.get("source") == "author_defined" else 1, str(value.get("scene_id") or ""))):
        description = re.sub(r"\s+", " ", str(item.get("description") or "").strip())
        if not description:
            continue
        key = (str(item.get("source") or ""), _normalize_name(description))
        if key in seen:
            continue
        sources.append(
            {
                "source": str(item.get("source") or ""),
                "scene_id": str(item.get("scene_id") or ""),
                "description": description,
            }
        )
        seen.add(key)
    return sources


def _name_parts(name: str) -> dict[str, Any]:
    clean = _strip_title(name)
    if _contains_cjk(clean):
        return {
            "full": clean,
            "surname_or_prefix": clean[:1] if len(clean) >= 2 else "",
            "given_or_suffix": clean[1:] if len(clean) >= 2 else clean,
        }
    parts = _latin_tokens(clean)
    return {
        "full": clean,
        "first": parts[0] if parts else "",
        "last": parts[-1] if len(parts) >= 2 else "",
        "tokens": parts,
    }


def _alias_index(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entity in entities:
        names = [entity.get("canonical_name", ""), *_as_list(entity.get("aliases"))]
        for name in names:
            key = _normalize_name(str(name))
            if key:
                index[key] = entity
    return index


def _resolve_entity(name: str, alias_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for variant in resolve_name_variants(name):
        match = alias_index.get(_normalize_name(variant))
        if match:
            return match
    return alias_index.get(_normalize_name(name))


def _chinese_variants(name: str) -> set[str]:
    clean = re.sub(r"\s+", "", name)
    variants = {clean}
    if _looks_like_chinese_person_name(clean):
        variants.add(clean[1:])
    return variants


def _looks_like_chinese_person_name(name: str) -> bool:
    return (
        3 <= len(name) <= 4
        and _has_likely_chinese_surname(name)
        and not any(term in name for term in _NON_PERSON_NAME_TERMS)
        and not any(name.endswith(suffix) for suffix in _ROLE_SUFFIXES)
    )


def _has_likely_chinese_surname(name: str) -> bool:
    return name[:2] in _COMPOUND_CHINESE_SURNAMES or name[:1] in _COMMON_CHINESE_SURNAMES


def _choose_entity_type(types: list[str]) -> str:
    normalized_types = [normalize_entity_type(entity_type) for entity_type in types]
    counts = {entity_type: normalized_types.count(entity_type) for entity_type in set(normalized_types)}
    if counts.get("character", 0) and counts.get("group", 0):
        return "group" if counts["group"] >= counts["character"] else "character"
    return sorted(counts, key=lambda key: (-counts[key], _TYPE_PRIORITY.index(key) if key in _TYPE_PRIORITY else 999, key))[0] if counts else "concept"


def _latin_variants(name: str) -> set[str]:
    parts = _latin_tokens(name)
    variants = {_normalize_space(name)}
    if len(parts) >= 2:
        variants.add(" ".join(parts))
        variants.add(" ".join(reversed(parts)))
        variants.add(parts[0])
        variants.add(parts[-1])
        variants.add(f"{parts[-1]}, {parts[0]}")
    return variants


def _latin_tokens(name: str) -> list[str]:
    return [part for part in re.split(r"[\s,\-.]+", _strip_title(name).strip()) if part]


def _strip_title(name: str) -> str:
    clean = str(name or "").strip()
    for title in _TITLE_PREFIXES:
        if clean.lower().startswith(title.lower() + " "):
            return clean[len(title) :].strip()
        if clean.startswith(title):
            return clean[len(title) :].strip()
    return clean


def _normalize_space(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip())


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s,\-.，。:：/（）()]+", "", str(name or "").lower())


def _is_deictic_name(name: object) -> bool:
    clean = _normalize_space(str(name or "")).strip().lower()
    if not clean:
        return False
    compact = _normalize_name(clean)
    if compact in _DEICTIC_TERMS:
        return True
    return any(clean.startswith(prefix) for prefix in _DEICTIC_PREFIXES)


def _embedded_alnum_code_key(name: object) -> str:
    text = str(name or "")
    matches = re.findall(r"[A-Za-z]*\d+[A-Za-z]*", text)
    return matches[-1].lower() if matches else ""


def _role_entity_key(name: object) -> str:
    clean = str(name or "").strip()
    clean = re.sub(r"^(为首的|那位|这位|一个|一位|几位|多位|那个|这个)", "", clean)
    clean = re.sub(r"(们)$", "", clean)
    normalized = _normalize_name(clean)
    if normalized in {"印度科学家", "科学家", "科研人员", "研究人员"}:
        return normalized
    return ""


def _contains_cjk(name: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", name or ""))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
