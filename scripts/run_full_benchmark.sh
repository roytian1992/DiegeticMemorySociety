#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-data/raw/流浪地球2剧本.json}"
CONFIG_PATH="${CONFIG_PATH:-configs/local_config.yaml}"
RUN_ID="${RUN_ID:-we2_full_$(date +%Y%m%d_%H%M%S)}"
SCENE_START="${SCENE_START:-1}"
SCENE_LIMIT="${SCENE_LIMIT:-}"
SCENE_TASK_CONCURRENCY="${SCENE_TASK_CONCURRENCY:-3}"
MAX_CHUNK_UNITS="${MAX_CHUNK_UNITS:-800}"
COLLECTION_NAME="${COLLECTION_NAME:-dms_retrieval_documents_bge_m3}"
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-}"

PREPARE_DIR="${PREPARE_DIR:-runs/benchmark_prepare/${RUN_ID}}"
ORDERED_RUN_DIR="${ORDERED_RUN_DIR:-runs/scene_ordered/${RUN_ID}}"
DB_PATH="${DB_PATH:-runs/assets/${RUN_ID}.sqlite}"
CHROMA_DIR="${CHROMA_DIR:-runs/assets/${RUN_ID}_chroma_bge_m3}"
BENCHMARK_DIR="${BENCHMARK_DIR:-runs/benchmark/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}"

mkdir -p "$LOG_DIR" "$(dirname "$DB_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[dms] started $(date -Is)"
echo "[dms] run_id=${RUN_ID}"
echo "[dms] python=${PYTHON_BIN}"
echo "[dms] script=${SCRIPT_PATH}"
echo "[dms] prepare_dir=${PREPARE_DIR}"
echo "[dms] ordered_run_dir=${ORDERED_RUN_DIR}"
echo "[dms] db_path=${DB_PATH}"
echo "[dms] chroma_dir=${CHROMA_DIR}"
echo "[dms] benchmark_dir=${BENCHMARK_DIR}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[dms] missing script file: ${SCRIPT_PATH}" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[dms] missing local model config: ${CONFIG_PATH}" >&2
  exit 1
fi

prepare_args=(
  -m dms.cli prepare-writing-benchmark
  "$SCRIPT_PATH"
  --output-dir "$PREPARE_DIR"
  --model-config "$CONFIG_PATH"
  --llm-section llm
  --embedding-section embedding
  --extraction-output-root "$ORDERED_RUN_DIR"
  --run-extraction
  --start "$SCENE_START"
  --scene-task-concurrency "$SCENE_TASK_CONCURRENCY"
  --max-chunk-units "$MAX_CHUNK_UNITS"
  --db-path "$DB_PATH"
  --chroma-dir "$CHROMA_DIR"
  --collection-name "$COLLECTION_NAME"
  --no-dry-run
  --overwrite
)

if [[ -n "$SCENE_LIMIT" ]]; then
  prepare_args+=(--limit "$SCENE_LIMIT")
fi

echo "[dms] preparing full assets"
PYTHONPATH=src "$PYTHON_BIN" "${prepare_args[@]}"

benchmark_args=(
  -m dms.cli run-writing-benchmark
  "$SCRIPT_PATH"
  --db-path "$DB_PATH"
  --chroma-dir "$CHROMA_DIR"
  --collection-name "$COLLECTION_NAME"
  --output-dir "$BENCHMARK_DIR"
  --model-config "$CONFIG_PATH"
  --llm-section llm
  --writing-llm-section writing_llm
  --embedding-section embedding
  --eligibility-dir "${PREPARE_DIR}/eligibility"
  --overwrite
)

if [[ -n "$BENCHMARK_LIMIT" ]]; then
  benchmark_args+=(--limit "$BENCHMARK_LIMIT")
else
  benchmark_args+=(--all-targets)
fi

echo "[dms] running writing benchmark"
PYTHONPATH=src "$PYTHON_BIN" "${benchmark_args[@]}"

echo "[dms] complete $(date -Is)"
