from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from dms.benchmark import WritingBenchmarkRunConfig, run_writing_benchmark
from dms.evaluation import build_scene_eligibility_splits
from dms.scripts.wandering_earth import load_script_scenes


DEFAULT_SCRIPT = Path("data/raw/流浪地球2剧本.json")
DEFAULT_DB = Path("runs/assets/we2_scene12345_7types.sqlite")
DEFAULT_CHROMA = Path("runs/assets/we2_scene12345_7types_chroma_bge_m3")
DEFAULT_COLLECTION = "dms_retrieval_documents_bge_m3"
DEFAULT_BENCHMARK_DIR = Path("runs/benchmark")


def build_app(
    *,
    script_path: Path = DEFAULT_SCRIPT,
    db_path: Path = DEFAULT_DB,
    chroma_dir: Path = DEFAULT_CHROMA,
    collection_name: str = DEFAULT_COLLECTION,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    model_config: Path = Path("configs/local_config.yaml"),
) -> gr.Blocks:
    scenes = load_script_scenes(script_path)
    scene_choices = [f"{scene.scene_id} | {scene.title}" for scene in scenes]

    with gr.Blocks(title="Diegetic Memory Society") as app:
        gr.Markdown("# Diegetic Memory Society")
        gr.Markdown("Scene-level memory retrieval, character attribute cards, social simulation, writing, and evaluation.")

        with gr.Tab("Benchmark Overview"):
            bench_path = gr.Textbox(value=str(benchmark_dir), label="Benchmark directory")
            overview_button = gr.Button("Refresh Overview")
            overview_table = gr.Dataframe(label="Target Results", interactive=False)
            overview_summary = gr.JSON(label="Summary")
            overview_button.click(_overview, inputs=[bench_path], outputs=[overview_table, overview_summary])

        with gr.Tab("Scene Inspector"):
            scene_select = gr.Dropdown(choices=scene_choices, value=scene_choices[0] if scene_choices else None, label="Scene")
            run_dir = gr.Textbox(value=str(benchmark_dir), label="Benchmark directory")
            inspect_button = gr.Button("Load Scene Artifacts")
            with gr.Row():
                scene_meta = gr.JSON(label="Scene")
                score_meta = gr.JSON(label="Scores")
            with gr.Row():
                sparse_intent = gr.Textbox(label="Sparse Intent", lines=3)
                detailed_intent = gr.Textbox(label="Detailed Intent", lines=3)
            with gr.Row():
                draft = gr.Textbox(label="Generated Draft", lines=8)
                reference = gr.Textbox(label="Reference Scene", lines=8)
            with gr.Row():
                memory_packet = gr.Textbox(label="Memory Packet", lines=18)
                attribute_cards = gr.Textbox(label="Attribute Cards", lines=18)
            social_sim = gr.Textbox(label="Social Simulation", lines=22)
            inspect_button.click(
                _inspect_scene,
                inputs=[scene_select, run_dir, gr.State(str(script_path))],
                outputs=[
                    scene_meta,
                    score_meta,
                    sparse_intent,
                    detailed_intent,
                    draft,
                    reference,
                    memory_packet,
                    attribute_cards,
                    social_sim,
                ],
            )

        with gr.Tab("Run One Scene"):
            gr.Markdown("Runs sparse/detailed intent extraction, retrieval, attribute cards, social simulation, writing, and detailed-intent evaluation for one scene.")
            with gr.Row():
                run_scene = gr.Dropdown(choices=scene_choices, value=scene_choices[5] if len(scene_choices) > 5 else (scene_choices[0] if scene_choices else None), label="Target Scene")
                run_output = gr.Textbox(value=str(benchmark_dir / "ui_single_scene"), label="Output directory")
            with gr.Row():
                run_db = gr.Textbox(value=str(db_path), label="SQLite asset DB")
                run_chroma = gr.Textbox(value=str(chroma_dir), label="Chroma directory")
            with gr.Row():
                run_collection = gr.Textbox(value=collection_name, label="Chroma collection")
                run_config = gr.Textbox(value=str(model_config), label="Model config")
            run_button = gr.Button("Run Scene")
            run_status = gr.JSON(label="Run Summary")
            run_draft = gr.Textbox(label="Generated Draft", lines=8)
            run_social = gr.Textbox(label="Social Simulation", lines=18)
            run_button.click(
                _run_one_scene,
                inputs=[run_scene, run_output, run_db, run_chroma, run_collection, run_config, gr.State(str(script_path))],
                outputs=[run_status, run_draft, run_social],
            )

        with gr.Tab("Eligibility"):
            eligibility_output = gr.Textbox(value=str(benchmark_dir / "eligibility_preview"), label="Output directory")
            eligibility_button = gr.Button("Build / Refresh Eligibility")
            eligibility_summary = gr.JSON(label="Eligibility Summary")
            eligibility_button.click(
                _build_eligibility,
                inputs=[eligibility_output, gr.State(str(script_path))],
                outputs=[eligibility_summary],
            )

    return app


def _scene_id(choice: str | None) -> str:
    return str(choice or "").split("|", 1)[0].strip()


def _overview(benchmark_dir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(benchmark_dir)
    summary = _read_json(root / "summary.json")
    rows: list[dict[str, Any]] = []
    metrics_path = root / "metrics.jsonl"
    if metrics_path.is_file():
        for row in _read_jsonl(metrics_path):
            rows.append(_flatten_target_result(row))
    elif (root / "targets").is_dir():
        for target_summary in sorted((root / "targets").glob("*/summary.json")):
            rows.append(_flatten_target_result(_read_json(target_summary)))
    return pd.DataFrame(rows), summary


def _inspect_scene(
    scene_choice: str | None,
    benchmark_dir: str,
    script_path: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str, str, str, str]:
    scene_id = _scene_id(scene_choice)
    target_dir = Path(benchmark_dir) / "targets" / scene_id
    scene = _load_scene(Path(script_path), scene_id)
    summary = _read_json(target_dir / "summary.json")
    return (
        scene.to_dict() if scene else {"scene_id": scene_id, "error": "scene not found"},
        summary.get("metrics", {}),
        _read_text(target_dir / "intents" / "sparse" / "writing_intent.txt"),
        _read_text(target_dir / "intents" / "detailed" / "writing_intent.txt"),
        _read_text(target_dir / "writing" / "draft.md"),
        scene.content if scene else "",
        _read_text(target_dir / "memory_packet.md"),
        _read_text(target_dir / "attribute_cards" / "attribute_cards.md"),
        _read_text(target_dir / "social_simulation" / "social_simulation.md"),
    )


def _run_one_scene(
    scene_choice: str | None,
    output_dir: str,
    db_path: str,
    chroma_dir: str,
    collection_name: str,
    model_config: str,
    script_path: str,
) -> tuple[dict[str, Any], str, str]:
    scene_id = _scene_id(scene_choice)
    summary = run_writing_benchmark(
        WritingBenchmarkRunConfig(
            script_path=Path(script_path),
            db_path=Path(db_path),
            chroma_dir=Path(chroma_dir),
            output_dir=Path(output_dir),
            model_config_path=Path(model_config),
            target_scene_ids=(scene_id,),
            limit=1,
            collection_name=collection_name,
            overwrite=True,
        )
    )
    target_dir = Path(output_dir) / "targets" / scene_id
    return (
        summary,
        _read_text(target_dir / "writing" / "draft.md"),
        _read_text(target_dir / "social_simulation" / "social_simulation.md"),
    )


def _build_eligibility(output_dir: str, script_path: str) -> dict[str, Any]:
    return build_scene_eligibility_splits(Path(script_path), Path(output_dir))


def _flatten_target_result(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    generated = metrics.get("generated") or {}
    reference = metrics.get("reference") or {}
    deltas = metrics.get("deltas") or {}
    return {
        "scene_id": row.get("scene_id"),
        "title": row.get("title"),
        "status": row.get("status"),
        "gen_intent": generated.get("writing_intent_consistency"),
        "gen_quality": generated.get("writing_quality"),
        "gen_faith": generated.get("memory_faithfulness"),
        "gen_overall": generated.get("overall"),
        "ref_overall": reference.get("overall"),
        "delta_intent": deltas.get("writing_intent_consistency"),
        "delta_overall": deltas.get("overall"),
        "entities": (row.get("counts") or {}).get("retrieved_entities"),
        "memories": (row.get("counts") or {}).get("retrieved_memories"),
    }


def _load_scene(script_path: Path, scene_id: str):
    for scene in load_script_scenes(script_path):
        if scene.scene_id == scene_id:
            return scene
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dms-gradio")
    parser.add_argument("--script-path", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--model-config", type=Path, default=Path("configs/local_config.yaml"))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    app = build_app(
        script_path=args.script_path,
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        benchmark_dir=args.benchmark_dir,
        model_config=args.model_config,
    )
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
