#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/preprocess${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${PROCESSED_DATA_LOG_DIR:-$REPO_ROOT/logs/preprocess/processed_data}"
RUN_ID="${PROCESSED_DATA_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
RUN_SUMMARY_LOG="$RUN_LOG_DIR/preprocess_summary.log"
MIRROR_TO_TERMINAL="${PROCESSED_DATA_LOG_MIRROR:-0}"
FORCE_REBUILD="${PROCESSED_DATA_FORCE_REBUILD:-0}"

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

processed_train_dev_ready() {
  local dataset="$1"
  local processed_dir="./data/processed_data/${dataset}"
  [[ -f "${processed_dir}/train.jsonl" ]] && [[ -f "${processed_dir}/dev.jsonl" ]]
}

processed_subsampled_ready() {
  local dataset="$1"
  local processed_dir="./data/processed_data/${dataset}"
  [[ -f "${processed_dir}/dev_subsampled.jsonl" ]] && [[ -f "${processed_dir}/test_subsampled.jsonl" ]]
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

run_processed_data_preprocess() {
  local dataset="$1"

  case "$dataset" in
    hotpotqa)
      "$PYTHON_BIN" ./preprocess/process_hotpotqa.py
      ;;
    2wikimultihopqa)
      "$PYTHON_BIN" ./preprocess/process_2wikimultihopqa.py
      ;;
    musique)
      "$PYTHON_BIN" ./preprocess/process_musique.py
      ;;
    *)
      echo "Unsupported dataset for processed data preprocessing: ${dataset}" >&2
      return 1
      ;;
  esac
}

run_processed_data_sampling() {
  local dataset="$1"
  "$PYTHON_BIN" ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name "$dataset" --set_name dev
  "$PYTHON_BIN" ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name "$dataset" --set_name test
}

prepare_dataset() {
  local dataset="$1"

  run_step_if_needed "$dataset" preprocess_processed_data processed_train_dev_ready "$dataset" \
    run_processed_data_preprocess "$dataset"

  run_step_if_needed "$dataset" sample_processed_data processed_subsampled_ready "$dataset" \
    run_processed_data_sampling "$dataset"
}

note "Processed-data logs will be written under ${RUN_LOG_DIR}"
note "This script only creates data/processed_data for train/inference. Retrieval DB build is handled separately by script/download/build.sh."
note "Using PYTHONPATH=${PYTHONPATH}"
note "Resume mode is $( [[ "$FORCE_REBUILD" == "1" ]] && printf 'disabled' || printf 'enabled' ); completed steps will $( [[ "$FORCE_REBUILD" == "1" ]] && printf 'rerun' || printf 'be skipped' )."

prepare_dataset hotpotqa
prepare_dataset musique
prepare_dataset 2wikimultihopqa

note "All processed-data datasets finished successfully"
