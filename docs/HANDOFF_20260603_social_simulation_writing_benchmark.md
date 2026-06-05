# Handoff 2026-06-03: Social Simulation And Writing Benchmark

## Scope

This handoff covers the current DiegeticMemorySociety writing benchmark work:

- three writing-task artifacts: `social_simulation_intent`, `writing_intent`, `writing_spec`
- source-isolated social simulation for held-out target scenes
- ASIP v0 algorithmic social planner and writer-facing `writer_packet.md`
- previous-scene context integration for writing
- writing quick-eval safeguards and benchmark metric plumbing
- current trusted scene-6 smoke results and known unusable historical outputs

Repository root:

```text
/vepfs-mlp2/c20250513/241404044/users/roytian/DiegeticMemorySociety
```

## Current State

Git state checked on 2026-06-03:

```text
main...origin/main
```

Latest pushed commits:

```text
f412b1e docs: document writing benchmark intent boundaries
5b232db feat: add source-isolated ASIP writing benchmark flow
e13e4ce docs: add social simulation research notes
```

Tracked working tree is clean. The local experiment log `docs/experiment_log_20260529.md` is ignored by `.gitignore` and contains extra maintenance notes not committed to Git.

Post-commit verification already passed:

```text
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m pytest -q
# 166 passed, 5 warnings

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m compileall -q src tests
# passed

git diff --check
# passed
```

Warnings were the known `websockets.legacy` deprecation and Chroma legacy hash embedding warnings.

## Trusted Results

These files were read directly before writing this handoff.

### ASIP Social Simulation Smoke

Path:

```text
runs/dev/social_sim_scene6_asip_v0_20260601_151500/summary.json
runs/dev/social_sim_scene6_asip_v0_20260601_151500/algorithmic_social_plan.json
runs/dev/social_sim_scene6_asip_v0_20260601_151500/writer_packet.md
runs/dev/social_sim_scene6_asip_v0_20260601_151500/writer_packet_verification.json
```

Usable metrics from `summary.json`:

```json
{
  "character_count": 2,
  "pressure_graph_edge_count": 2,
  "candidate_action_count": 12,
  "selected_beat_count": 2,
  "selected_sequence_score": 0.88,
  "scene_problem_coverage": 1,
  "relationship_coverage": 1,
  "memory_support_rate": 1,
  "candidate_memory_ref_count": 48,
  "hard_violation_count": 0,
  "soft_warning_count": 4,
  "awkward_phrase_risk_count": 0,
  "writer_packet_action_guidance_count": 4,
  "writer_packet_dialogue_posture_count": 2,
  "writer_packet_avoid_phrase_count": 0,
  "raw_simulation_warning_type_count": 1
}
```

Writer-packet verification metrics:

```json
{
  "hard_violation_count": 0,
  "soft_warning_count": 0,
  "therapy_phrase_risk_count": 0,
  "unsupported_role_risk_count": 0,
  "final_dialogue_like_count": 0,
  "missing_not_final_dialogue_flag_count": 0
}
```

Note: this run predates the final `source_isolation` summary field, so do not use this run alone as proof that source isolation is visible in output metadata. Use the current code/tests for that guarantee.

### Best Current Scene-6 Writing Smoke

Path:

```text
runs/dev/asip_writer_packet_writing_scene6_promptopt_anchors_20260601_221000/quick_eval.json
runs/dev/asip_writer_packet_writing_scene6_promptopt_anchors_20260601_221000/evaluation_relaxed_spec_v2/summary.json
```

Quick eval:

```json
{
  "body_non_ws_chars": 164,
  "request_anchors": ["刘培强", "张鹏", "J20C"],
  "missing_request_anchors": [],
  "ref_ids_present": [],
  "writer_packet_artifact_terms_present": [],
  "dialogue_risk_phrases_present": []
}
```

Evaluation:

```json
{
  "generated_overall": 1,
  "reference_overall": 0.9333,
  "deltas": {
    "writing_intent_consistency": 0,
    "writing_quality": 0,
    "memory_faithfulness": 0.2,
    "overall": 0.0667
  }
}
```

### Earlier Valid Regeneration Comparisons

Social-simulation + writing intent + previous scene:

```text
runs/dev/social_writing_intent_prev_context_regen_scene6_20260601_131200/evaluation_relaxed_spec_v2/summary.json
```

Generated overall and reference overall were both `1`, with all deltas `0`.

Non-social comparison:

```text
runs/dev/non_social_writing_intent_prev_context_regen_scene6_20260601_133000/evaluation_relaxed_spec_v2/summary.json
```

Generated overall and reference overall were both `1`, with all deltas `0`. Use this only as a score comparison, not as a style exemplar; see the warning below.

## Untrusted Or Failed Results

- `runs/dev/three_intents_scene6_20260601_103137/targets/scene_0006/summary.json` uses the legacy `reference_scene_spec` naming and an overly strict spec that still requires `UEG基地`, `非洲中部`, and exact flight mechanics. Treat it as history, not the current evaluation policy.
- `runs/dev/asip_writer_packet_writing_scene6_promptopt_20260601_220000/quick_eval.json` exposed `missing_request_anchors=["J20C"]`. Do not use this draft as a final exemplar.
- `runs/dev/non_social_writing_intent_prev_context_regen_scene6_20260601_133000` scored well, but the draft contains `别跟地球赌气`, which was manually judged awkward. Current quick-eval risk phrases include this phrase; do not use that wording in final examples.
- Full raw `social_simulation.md` is a debug artifact. Writing should consume `social_simulation/writer_packet.md`, not the full raw simulation markdown.
- Any social simulation run that exposes `content`, `unit_json`, `target_scene_text`, `reference_text`, `writing_spec`, or `reference_scene_spec` to the social simulation prompt is contaminated for source-isolation claims.

## Important Paths

### Code

```text
src/dms/benchmark.py
src/dms/intent_levels.py
src/dms/writing.py
src/dms/workflow.py
src/dms/simulation/social.py
src/dms/simulation/algorithmic.py
src/dms/simulation/verification.py
src/dms/simulation/formatting.py
src/dms/ui/gradio_app.py
src/dms/cli.py
src/dms/prompts/loader.py
```

### Prompts And Task Settings

```text
task_specs/prompts/dms/social_simulation_intent.yaml
task_specs/prompts/dms/writing_intent.yaml
task_specs/prompts/dms/writing_spec.yaml
task_specs/prompts/dms/character_social_simulation.yaml
task_specs/prompts/dms/social_simulation_coordinator.yaml
task_specs/prompts/dms/writing_generation.yaml
task_specs/prompts/dms/writing_generation_social.yaml
task_specs/prompts/dms/eval_intent_requirements.yaml
task_specs/task_settings/social_simulation_intent_task.json
task_specs/task_settings/writing_intent_task.json
task_specs/task_settings/writing_spec_task.json
```

Removed legacy files:

```text
task_specs/prompts/dms/writing_intent_sparse.yaml
task_specs/prompts/dms/writing_intent_detailed.yaml
task_specs/task_settings/writing_intent_sparse_task.json
task_specs/task_settings/writing_intent_detailed_task.json
```

### Tests

```text
tests/test_intent_levels.py
tests/test_prompt_loader.py
tests/test_social_simulation.py
tests/test_social_simulation_verification.py
tests/test_writing_generation.py
tests/test_writing_benchmark.py
tests/test_writing_e2e_workflow.py
tests/test_writing_evaluation.py
tests/test_gradio_app.py
```

### Documentation

```text
README.md
docs/ARCHITECTURE.md
docs/EVALUATION_DESIGN.md
docs/RETRIEVAL_DESIGN.md
docs/SOCIAL_SIMULATION_DESIGN.md
docs/SOCIAL_SIMULATION_RESEARCH.md
docs/SOCIAL_SIMULATION_PAPER_REFERENCES.md
docs/experiment_log_20260529.md
```

`docs/experiment_log_20260529.md` is ignored by Git and is local maintenance state.

### Inputs And Assets

Checked paths:

```text
configs/local_config.yaml
data/raw/流浪地球2剧本.json
runs/assets/we2_scene12345_7types.sqlite
runs/assets/we2_scene12345_7types_chroma_bge_m3
runs/assets/we2_scene12345_7types_chroma_bge_m3/chroma.sqlite3
```

Default Chroma collection for the current asset set:

```text
dms_retrieval_documents_bge_m3
```

## Important Scripts

CLI entrypoint:

```text
src/dms/cli.py
```

Core programmatic entrypoints:

```text
src/dms/benchmark.py: WritingBenchmarkRunConfig, run_writing_benchmark
src/dms/workflow.py: WritingE2EConfig, run_writing_e2e
src/dms/simulation/social.py: SocialSimulationConfig, run_social_simulation
src/dms/simulation/algorithmic.py: build_algorithmic_social_plan
src/dms/simulation/verification.py: verify_social_simulation, verify_writer_packet, detect_text_risks
src/dms/writing.py: SocialWritingGenerationConfig, generate_writing_with_social_simulation_client
```

Relevant CLI subcommands:

```text
build-memory-packet
build-entity-attribute-cards
run-social-simulation
generate-writing-social
run-writing-e2e
prepare-writing-benchmark
run-writing-benchmark
launch-ui
```

## What Was Tried

- Replaced the old sparse/detailed intent split with three explicit artifacts:
  `social_simulation_intent`, `writing_intent`, and `writing_spec`.
- Tested social simulation with less target-scene information than normal writing intent.
- Added previous-scene context to writing as auxiliary continuity, capped at 800 non-whitespace chars; it does not replace `writing_intent`.
- Relaxed the evaluation spec so `writing_spec` checks core scene function rather than exact target-scene choreography.
- Built ASIP v0: social state graph, pressure graph, typed action candidates, candidate sequence search/reranking, writer packet, and verification.
- Moved writing to consume `writer_packet.md` instead of raw `social_simulation.md`.
- Added quick-eval checks for missing request anchors, M/R ref leakage, writer-packet artifact leakage, and awkward/therapy-like dialogue phrases.
- Ran and pushed three commits to `origin/main`.

## What Was Learned

- `writing_spec` should remain evaluation-only. If it is fed into writing/social simulation, the task becomes contaminated.
- `social_simulation_intent` should be lower information than `writing_intent`; it is enough to set up the interaction, not the exact target beat list.
- Social simulation improved interpersonal pressure and active cockpit behavior compared with non-social generation, but only when the writer consumes the cleaned writer packet.
- Raw social simulation output can contain debug warnings or overly literal dialogue-like text. The writer packet plus `verify_writer_packet` is the safe handoff surface.
- Previous-scene context is useful for local continuity, but prompt wording must explicitly prevent it from replacing request anchors.
- Evaluator requirements should be based on `Required ...` fields in `writing_spec`; broad `Scene purpose` should guide weighting, not create extra hard constraints.
- The phrase `别跟地球赌气` is a known awkward output pattern and is now guarded in quick eval.

## Recommended Next Steps

1. Run a fresh benchmark target from the pushed code, not only historical `runs/dev` artifacts, so new summaries include the latest canonical `writing_spec` and source-isolation metadata.
2. Use scene 6 first because previous smoke tests and expected anchors are known.
3. Inspect `targets/scene_0006/social_simulation/summary.json`, `writer_packet.md`, `writer_packet_verification.json`, `writing/quick_eval.json`, and `evaluation/summary.json`.
4. If scene 6 is stable, run a small multi-scene benchmark with `--limit 3` and compare aggregate metrics plus manual drafts.
5. Add a social-simulation quality report that separates algorithmic metrics from final writing metrics.
6. Consider making a formal result table from `metrics.jsonl`, but only after a fresh current-code run.

## Command Templates

Run one fresh scene-6 benchmark target:

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/DiegeticMemorySociety

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m dms.cli run-writing-benchmark \
  data/raw/流浪地球2剧本.json \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --target-scene-id scene_0006 \
  --limit 1 \
  --output-dir runs/dev/current_scene6_benchmark_$(date +%Y%m%d_%H%M%S) \
  --overwrite
```

Run intent-only extraction for one scene:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m dms.cli run-writing-benchmark \
  data/raw/流浪地球2剧本.json \
  --db-path runs/assets/we2_scene12345_7types.sqlite \
  --chroma-dir runs/assets/we2_scene12345_7types_chroma_bge_m3 \
  --collection-name dms_retrieval_documents_bge_m3 \
  --target-scene-id scene_0006 \
  --limit 1 \
  --intent-only \
  --output-dir runs/dev/current_scene6_intents_$(date +%Y%m%d_%H%M%S) \
  --overwrite
```

Run regression before changing benchmark code:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m pytest -q
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m compileall -q src tests
git diff --check
```

Open the Gradio UI:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/screenplay/bin/python -m dms.cli launch-ui \
  --benchmark-dir runs/benchmark \
  --server-name 127.0.0.1 \
  --server-port 7860
```

## Handoff Caveats

- Historical runs under `runs/dev` are useful for comparison, but the best next action is a fresh current-code benchmark run.
- Do not use `writing_spec` or target scene text in social simulation prompts.
- Do not evaluate final writing against `writing_intent`; benchmark evaluation should use `writing_spec`.
- Do not let previous-scene context replace explicit anchors in the writing request.
- The current full regression passed before this handoff, but any future prompt/model config change should be followed by at least focused tests plus one real smoke run.
