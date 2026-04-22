#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${BUILD_LOG_DIR:-$REPO_ROOT/logs/build}"
RUN_ID="${BUILD_LOG_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
RUN_SUMMARY_LOG="$RUN_LOG_DIR/build_summary.log"
MIRROR_TO_TERMINAL="${BUILD_LOG_MIRROR:-0}"

mkdir -p "$RUN_LOG_DIR"

note() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$RUN_SUMMARY_LOG"
}

run_step() {
  local dataset="$1"
  local step_name="$2"
  shift 2

  local log_file="$RUN_LOG_DIR/${dataset}__${step_name}.log"
  local status=0

  note "Running ${dataset}/${step_name}. Log: ${log_file}"

  set +e
  if [[ "$MIRROR_TO_TERMINAL" == "1" ]]; then
    "$@" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
  else
    "$@" >"$log_file" 2>&1
    status=$?
  fi
  set -e

  if [[ $status -ne 0 ]]; then
    note "Failed ${dataset}/${step_name} with exit code ${status}. Inspect: ${log_file}"
    return "$status"
  fi

  note "Finished ${dataset}/${step_name}"
}

build_dataset() {
  local dataset="$1"
  local passage_path="$2"
  local output_dir="$3"

  run_step "$dataset" preprocess_raw_data \
    "$PYTHON_BIN" -m source.run.preprocess_raw_data \
    --dataset_name "$dataset"

  run_step "$dataset" generate_passage_embeddings \
    "$PYTHON_BIN" -m source.run.generate_passage_embeddings \
    --model_name_or_path facebook/contriever-msmarco \
    --passages "$passage_path" \
    --output_dir "$output_dir" \
    --shard_id 0 \
    --num_shards 1 \
    --per_gpu_batch_size 1024

  run_step "$dataset" build_index \
    "$PYTHON_BIN" -m source.run.build_index \
    --output_dir "$output_dir"
}

note "Build logs will be written under ${RUN_LOG_DIR}"

build_dataset hotpotqa ./data/embed_ready_data/hotpotqa.tsv ./data/database/contriever_msmarco/hotpotqa
build_dataset musique ./data/embed_ready_data/musique.tsv ./data/database/contriever_msmarco/musique
build_dataset 2wikimultihopqa ./data/embed_ready_data/2wikimultihopqa.tsv ./data/database/contriever_msmarco/2wikimultihopqa

note "All datasets finished successfully"
