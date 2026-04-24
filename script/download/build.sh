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
EMBED_CUDA_VISIBLE_DEVICES="${BUILD_CUDA_VISIBLE_DEVICES:-2,3,4,5}"
EMBED_BATCH_SIZE="${BUILD_EMBED_BATCH_SIZE:-128}"
EMBED_PASSAGE_MAX_LENGTH="${BUILD_EMBED_PASSAGE_MAX_LENGTH:-512}"
EMBED_ALLOC_CONF="${BUILD_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EMBED_SAVE_EVERY_BATCHES="${BUILD_EMBED_SAVE_EVERY_BATCHES:-64}"
FORCE_REBUILD="${BUILD_FORCE_REBUILD:-0}"

EMBED_CUDA_VISIBLE_DEVICES="${EMBED_CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a EMBED_GPU_ARRAY <<< "$EMBED_CUDA_VISIBLE_DEVICES"
EMBED_GPU_COUNT="${#EMBED_GPU_ARRAY[@]}"
EMBED_TOTAL_SHARDS="${BUILD_EMBED_TOTAL_SHARDS:-$EMBED_GPU_COUNT}"

if [[ "$EMBED_GPU_COUNT" -eq 0 ]]; then
  echo "BUILD_CUDA_VISIBLE_DEVICES must contain at least one GPU id." >&2
  exit 1
fi

mkdir -p "$RUN_LOG_DIR"

note() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$RUN_SUMMARY_LOG"
}

describe_path_location() {
  local label="$1"
  local path="$2"
  local resolved_path

  resolved_path="$(resolve_path_for_reporting "$path")"

  if [[ "$resolved_path" == /docker || "$resolved_path" == /docker/* ]]; then
    note "${label}: path=${path} realpath=${resolved_path} [/docker]"
  else
    note "${label}: path=${path} realpath=${resolved_path} [non-/docker]"
  fi
}

resolve_path_for_reporting() {
  local path="$1"
  if [[ -e "$path" ]]; then
    realpath "$path"
  else
    printf '%s\n' "$path"
  fi
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

preprocess_ready() {
  local passage_path="$1"
  [[ -s "$passage_path" ]] || return 1
  [[ $(wc -l < "$passage_path") -gt 1 ]]
}

shard_embedding_marker_path() {
  local output_dir="$1"
  local shard_id="$2"
  printf '%s/embedding_shard_%02d.complete.json\n' "$output_dir" "$shard_id"
}

shard_embeddings_ready() {
  local output_dir="$1"
  local shard_id="$2"
  local marker_path
  marker_path="$(shard_embedding_marker_path "$output_dir" "$shard_id")"
  [[ -f "$marker_path" ]]
}

embedding_chunks_exist() {
  local output_dir="$1"
  compgen -G "${output_dir}/embeddings_[0-9][0-9]*" > /dev/null
}

embeddings_ready() {
  local output_dir="$1"
  local num_shards="$2"
  local shard_id
  for (( shard_id=0; shard_id<num_shards; shard_id++ )); do
    shard_embeddings_ready "$output_dir" "$shard_id" || return 1
  done
}

index_ready() {
  local output_dir="$1"
  [[ -f "${output_dir}/index.faiss" ]] \
    && [[ -f "${output_dir}/faiss_id_to_docstore_id.pkl" ]] \
    && [[ -f "${output_dir}/docstore.db" ]]
}

run_step_if_needed() {
  local dataset="$1"
  local step_name="$2"
  local check_name="$3"
  local check_arg="$4"
  shift 4

  if [[ "$FORCE_REBUILD" != "1" ]] && "$check_name" "$check_arg"; then
    note "Skipping ${dataset}/${step_name}; output already exists."
    return 0
  fi

  run_step "$dataset" "$step_name" "$@"
}

run_embedding_workers() {
  local dataset="$1"
  local passage_path="$2"
  local output_dir="$3"
  local shard_base=0

  mkdir -p "$output_dir"
  echo "Parallel embedding config: GPUs=${EMBED_CUDA_VISIBLE_DEVICES}, total_shards=${EMBED_TOTAL_SHARDS}, save_every_batches=${EMBED_SAVE_EVERY_BATCHES}"

  while (( shard_base < EMBED_TOTAL_SHARDS )); do
    local -a pids=()
    local -a shard_ids=()
    local -a shard_logs=()
    local slot=0

    for gpu_id in "${EMBED_GPU_ARRAY[@]}"; do
      local shard_id=$(( shard_base + slot ))
      if (( shard_id >= EMBED_TOTAL_SHARDS )); then
        break
      fi

      if [[ "$FORCE_REBUILD" != "1" ]] && shard_embeddings_ready "$output_dir" "$shard_id"; then
        echo "Skipping shard ${shard_id}/${EMBED_TOTAL_SHARDS}; output already exists."
        slot=$(( slot + 1 ))
        continue
      fi

      local shard_log="$RUN_LOG_DIR/${dataset}__generate_passage_embeddings__shard_$(printf '%02d' "$shard_id").log"
      echo "Launching shard ${shard_id}/${EMBED_TOTAL_SHARDS} on GPU ${gpu_id}. Log: ${shard_log}"
      env \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        PYTORCH_CUDA_ALLOC_CONF="$EMBED_ALLOC_CONF" \
        "$PYTHON_BIN" -m source.run.generate_passage_embeddings \
        --model_name_or_path facebook/contriever-msmarco \
        --passages "$passage_path" \
        --output_dir "$output_dir" \
        --shard_id "$shard_id" \
        --num_shards "$EMBED_TOTAL_SHARDS" \
        --per_gpu_batch_size "$EMBED_BATCH_SIZE" \
        --passage_maxlength "$EMBED_PASSAGE_MAX_LENGTH" \
        --save_every_batches "$EMBED_SAVE_EVERY_BATCHES" \
        >"$shard_log" 2>&1 &

      pids+=("$!")
      shard_ids+=("$shard_id")
      shard_logs+=("$shard_log")
      slot=$(( slot + 1 ))
    done

    local wave_failed=0
    local idx
    for idx in "${!pids[@]}"; do
      if wait "${pids[$idx]}"; then
        echo "Finished shard ${shard_ids[$idx]}/${EMBED_TOTAL_SHARDS}."
      else
        echo "Failed shard ${shard_ids[$idx]}/${EMBED_TOTAL_SHARDS}. Inspect: ${shard_logs[$idx]}"
        wave_failed=1
      fi
    done

    if [[ "$wave_failed" -ne 0 ]]; then
      return 1
    fi

    shard_base=$(( shard_base + EMBED_GPU_COUNT ))
  done
}

run_embedding_if_needed() {
  local dataset="$1"
  local passage_path="$2"
  local output_dir="$3"
  local log_file="$RUN_LOG_DIR/${dataset}__generate_passage_embeddings.log"
  local status=0

  if [[ "$FORCE_REBUILD" != "1" ]] && embeddings_ready "$output_dir" "$EMBED_TOTAL_SHARDS"; then
    note "Skipping ${dataset}/generate_passage_embeddings; output already exists."
    return 0
  fi

  if [[ "$FORCE_REBUILD" != "1" ]] && embedding_chunks_exist "$output_dir"; then
    note "Skipping ${dataset}/generate_passage_embeddings; detected existing embedding chunk files under ${output_dir}. Proceeding to build_index."
    return 0
  fi

  note "Running ${dataset}/generate_passage_embeddings. Log: ${log_file}"

  set +e
  if [[ "$MIRROR_TO_TERMINAL" == "1" ]]; then
    run_embedding_workers "$dataset" "$passage_path" "$output_dir" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
  else
    run_embedding_workers "$dataset" "$passage_path" "$output_dir" >"$log_file" 2>&1
    status=$?
  fi
  set -e

  if [[ $status -ne 0 ]]; then
    note "Failed ${dataset}/generate_passage_embeddings with exit code ${status}. Inspect: ${log_file}"
    return "$status"
  fi

  note "Finished ${dataset}/generate_passage_embeddings"
}

build_dataset() {
  local dataset="$1"
  local passage_path="$2"
  local output_dir="$3"

  run_step_if_needed "$dataset" preprocess_raw_data preprocess_ready "$passage_path" \
    "$PYTHON_BIN" -m source.run.preprocess_raw_data \
    --dataset_name "$dataset"

  run_embedding_if_needed "$dataset" "$passage_path" "$output_dir"

  run_step_if_needed "$dataset" build_index index_ready "$output_dir" \
    "$PYTHON_BIN" -m source.run.build_index \
    --output_dir "$output_dir"
}

note "Build logs will be written under ${RUN_LOG_DIR}"
note "This script only builds retrieval DB artifacts under data/embed_ready_data and data/database."
note "Processed QA data under data/processed_data is prepared separately; use script/download/multihop_processed_data.sh if needed."
note "Embedding step will use CUDA_VISIBLE_DEVICES=${EMBED_CUDA_VISIBLE_DEVICES}, total_shards=${EMBED_TOTAL_SHARDS}, per_gpu_batch_size=${EMBED_BATCH_SIZE}, passage_maxlength=${EMBED_PASSAGE_MAX_LENGTH}, save_every_batches=${EMBED_SAVE_EVERY_BATCHES}"
note "Resume mode is $( [[ "$FORCE_REBUILD" == "1" ]] && printf 'disabled' || printf 'enabled' ); completed steps will $( [[ "$FORCE_REBUILD" == "1" ]] && printf 'rerun' || printf 'be skipped' )."
describe_path_location "build_log_dir" "$RUN_LOG_DIR"
describe_path_location "database_root" "$REPO_ROOT/data/database"
if [[ "$(resolve_path_for_reporting "$RUN_LOG_DIR")" != /docker* || "$(resolve_path_for_reporting "$REPO_ROOT/data/database")" != /docker* ]]; then
  note "Warning: build is still writing hot data outside /docker. Run: bash script/setup_local_hot_data.sh"
fi

build_dataset hotpotqa ./data/embed_ready_data/hotpotqa.tsv ./data/database/contriever_msmarco/hotpotqa
build_dataset musique ./data/embed_ready_data/musique.tsv ./data/database/contriever_msmarco/musique
build_dataset 2wikimultihopqa ./data/embed_ready_data/2wikimultihopqa.tsv ./data/database/contriever_msmarco/2wikimultihopqa

note "All datasets finished successfully"
