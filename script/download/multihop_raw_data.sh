#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEMP_ROOT="${RAW_DATA_TEMP_DIR:-$REPO_ROOT/.temp}"
LOG_ROOT="${RAW_DATA_LOG_DIR:-$REPO_ROOT/logs/download/raw_data}"
RUN_ID="${RAW_DATA_LOG_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
RUN_SUMMARY_LOG="$RUN_LOG_DIR/download_summary.log"
MIRROR_TO_TERMINAL="${RAW_DATA_LOG_MIRROR:-0}"
KEEP_TEMP="${RAW_DATA_KEEP_TEMP:-1}"
DOWNLOAD_RETRIES="${RAW_DATA_DOWNLOAD_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RAW_DATA_RETRY_SLEEP_SECONDS:-5}"
GDOWN_BIN="${GDOWN_BIN:-$(dirname "$PYTHON_BIN")/gdown}"
MUSIQUE_GDRIVE_ID="${MUSIQUE_GDRIVE_ID:-1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h}"

mkdir -p "$RUN_LOG_DIR" "$TEMP_ROOT" data/raw_data

note() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$RUN_SUMMARY_LOG"
}

on_exit() {
  local status=$?

  if [[ $status -eq 0 ]]; then
    note "All raw data download steps finished successfully"
    if [[ "$KEEP_TEMP" == "0" ]]; then
      rm -rf "$TEMP_ROOT"
      note "Removed temporary directory ${TEMP_ROOT}"
    else
      note "Temporary files kept at ${TEMP_ROOT}"
    fi
  else
    note "Raw data download failed. Inspect logs under ${RUN_LOG_DIR}"
    note "Temporary files kept at ${TEMP_ROOT} for debugging"
  fi
}

trap on_exit EXIT

run_step() {
  local step_name="$1"
  shift

  local log_file="$RUN_LOG_DIR/${step_name}.log"
  local status=0

  note "Running ${step_name}. Log: ${log_file}"

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
    note "Failed ${step_name} with exit code ${status}. Inspect: ${log_file}"
    return "$status"
  fi

  note "Finished ${step_name}"
}

run_if_missing() {
  local step_name="$1"
  local ready_fn="$2"
  local action_fn="$3"

  if "$ready_fn"; then
    note "Skipping ${step_name} because expected outputs already exist"
    return 0
  fi

  run_step "$step_name" "$action_fn"
}

retry_command() {
  local attempt=1

  while true; do
    if "$@"; then
      return 0
    fi

    if (( attempt >= DOWNLOAD_RETRIES )); then
      return 1
    fi

    echo "Attempt ${attempt}/${DOWNLOAD_RETRIES} failed. Retrying in ${RETRY_SLEEP_SECONDS}s..."
    sleep "$RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

have_all_files() {
  local path
  for path in "$@"; do
    if [[ ! -s "$path" ]]; then
      return 1
    fi
  done
}

have_matching_files() {
  compgen -G "$1" > /dev/null
}

download_url_to_file() {
  local url="$1"
  local output_path="$2"

  mkdir -p "$(dirname "$output_path")"
  rm -f "$output_path"

  retry_command wget \
    --retry-connrefused \
    --waitretry="$RETRY_SLEEP_SECONDS" \
    --timeout=60 \
    --tries=1 \
    -O "$output_path" \
    "$url"

  if [[ ! -s "$output_path" ]]; then
    echo "Downloaded file is missing or empty: $output_path"
    return 1
  fi
}

download_url_candidates_to_file() {
  local output_path="$1"
  shift

  local url
  local status=0

  for url in "$@"; do
    echo "Trying URL: $url"
    set +e
    download_url_to_file "$url" "$output_path"
    status=$?
    set -e

    if [[ $status -eq 0 ]]; then
      echo "Downloaded successfully from: $url"
      return 0
    fi

    rm -f "$output_path"
    echo "URL failed: $url"
  done

  echo "All candidate URLs failed for $output_path"
  return 1
}

ensure_gdown() {
  if [[ -x "$GDOWN_BIN" ]]; then
    echo "Using gdown binary at $GDOWN_BIN"
    return 0
  fi

  if command -v gdown > /dev/null 2>&1; then
    GDOWN_BIN="$(command -v gdown)"
    echo "Using gdown binary at $GDOWN_BIN"
    return 0
  fi

  "$PYTHON_BIN" -m pip install --disable-pip-version-check gdown

  if [[ -x "$(dirname "$PYTHON_BIN")/gdown" ]]; then
    GDOWN_BIN="$(dirname "$PYTHON_BIN")/gdown"
    echo "Using gdown binary at $GDOWN_BIN"
    return 0
  fi

  GDOWN_BIN="$(command -v gdown)"
  echo "Using gdown binary at $GDOWN_BIN"
}

hotpotqa_questions_ready() {
  have_all_files \
    data/raw_data/hotpotqa/hotpot_train_v1.1.json \
    data/raw_data/hotpotqa/hotpot_dev_distractor_v1.json
}

download_hotpotqa_questions() {
  mkdir -p data/raw_data/hotpotqa

  download_url_candidates_to_file \
    data/raw_data/hotpotqa/hotpot_train_v1.1.json \
    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

  download_url_candidates_to_file \
    data/raw_data/hotpotqa/hotpot_dev_distractor_v1.json \
    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

  hotpotqa_questions_ready
}

wikimultihopqa_ready() {
  have_all_files \
    data/raw_data/2wikimultihopqa/dev.json \
    data/raw_data/2wikimultihopqa/id_aliases.json \
    data/raw_data/2wikimultihopqa/test.json \
    data/raw_data/2wikimultihopqa/train.json
}

download_2wikimultihopqa() {
  mkdir -p data/raw_data/2wikimultihopqa

  download_url_to_file \
    "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1" \
    "$TEMP_ROOT/2wikimultihopqa.zip"

  unzip -o -j "$TEMP_ROOT/2wikimultihopqa.zip" -d data/raw_data/2wikimultihopqa -x "*.DS_Store"

  wikimultihopqa_ready
}

musique_ready() {
  have_all_files \
    data/raw_data/musique/dev_test_singlehop_questions_v1.0.json \
    data/raw_data/musique/musique_ans_v1.0_dev.jsonl \
    data/raw_data/musique/musique_ans_v1.0_test.jsonl \
    data/raw_data/musique/musique_ans_v1.0_train.jsonl \
    data/raw_data/musique/musique_full_v1.0_dev.jsonl \
    data/raw_data/musique/musique_full_v1.0_test.jsonl \
    data/raw_data/musique/musique_full_v1.0_train.jsonl
}

download_musique() {
  mkdir -p data/raw_data/musique
  ensure_gdown
  rm -f "$TEMP_ROOT/musique_v1.0.zip"

  retry_command "$GDOWN_BIN" \
    --id "$MUSIQUE_GDRIVE_ID" \
    -O "$TEMP_ROOT/musique_v1.0.zip"

  unzip -o -j "$TEMP_ROOT/musique_v1.0.zip" -d data/raw_data/musique -x "*.DS_Store"

  musique_ready
}

iirc_questions_ready() {
  have_all_files \
    data/raw_data/iirc/train.json \
    data/raw_data/iirc/dev.json
}

download_iirc_questions() {
  mkdir -p data/raw_data/iirc
  rm -rf "$TEMP_ROOT/iirc_train_dev"

  download_url_candidates_to_file \
    "$TEMP_ROOT/iirc_train_dev.tgz" \
    https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz

  tar -xzvf "$TEMP_ROOT/iirc_train_dev.tgz" -C "$TEMP_ROOT"
  cp -f "$TEMP_ROOT/iirc_train_dev/train.json" data/raw_data/iirc/train.json
  cp -f "$TEMP_ROOT/iirc_train_dev/dev.json" data/raw_data/iirc/dev.json

  iirc_questions_ready
}

iirc_corpus_ready() {
  have_all_files data/raw_data/iirc/context_articles.json
}

download_iirc_corpus() {
  mkdir -p data/raw_data/iirc "$TEMP_ROOT/iirc_corpus"
  rm -rf "$TEMP_ROOT/iirc_corpus"
  mkdir -p "$TEMP_ROOT/iirc_corpus"

  download_url_candidates_to_file \
    "$TEMP_ROOT/context_articles.tar.gz" \
    https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz \
    https://iirc-data.s3.us-west-2.amazonaws.com/context_articles.tar.gz

  tar -xzvf "$TEMP_ROOT/context_articles.tar.gz" -C "$TEMP_ROOT/iirc_corpus"
  cp -f "$TEMP_ROOT/iirc_corpus/context_articles.json" data/raw_data/iirc/context_articles.json

  iirc_corpus_ready
}

hotpotqa_corpus_ready() {
  have_matching_files "data/raw_data/hotpotqa/wikipedia-paragraphs/*/wiki_*.bz2" || \
    have_matching_files "data/raw_data/hotpotqa/wikpedia-paragraphs/*/wiki_*.bz2"
}

download_hotpotqa_corpus() {
  local extracted_dir="$TEMP_ROOT/hotpotqa_extract/enwiki-20171001-pages-meta-current-withlinks-abstracts"
  local target_dir="data/raw_data/hotpotqa/wikipedia-paragraphs"

  mkdir -p data/raw_data/hotpotqa
  rm -rf "$TEMP_ROOT/hotpotqa_extract"
  mkdir -p "$TEMP_ROOT/hotpotqa_extract" "$target_dir"

  download_url_to_file \
    https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 \
    "$TEMP_ROOT/wikipedia-paragraphs.tar.bz2"

  tar -xvf "$TEMP_ROOT/wikipedia-paragraphs.tar.bz2" -C "$TEMP_ROOT/hotpotqa_extract"
  cp -a "$extracted_dir/." "$target_dir/"

  hotpotqa_corpus_ready
}

note "Raw data download logs will be written under ${RUN_LOG_DIR}"

run_if_missing hotpotqa_questions hotpotqa_questions_ready download_hotpotqa_questions
run_if_missing 2wikimultihopqa wikimultihopqa_ready download_2wikimultihopqa
run_if_missing musique musique_ready download_musique
run_if_missing iirc_questions iirc_questions_ready download_iirc_questions
run_if_missing iirc_corpus iirc_corpus_ready download_iirc_corpus
run_if_missing hotpotqa_corpus hotpotqa_corpus_ready download_hotpotqa_corpus

# The resulting raw_data/ directory should look like:
# ── 2wikimultihopqa
# │   ├── dev.json
# │   ├── id_aliases.json
# │   ├── test.json
# │   └── train.json
# ├── hotpotqa
# │   ├── dev_random_20_single_hop_annotations.txt
# │   ├── wikipedia-paragraphs/
# │   ├──  ├── ...
# │   ├── hotpot_dev_distractor_v1.json
# │   └── train_random_20_single_hop_annotations.txt
# ├── iirc
# │   ├── context_articles.json
# │   ├── dev.json
# │   └── train.json
# └── musique
#     ├── dev_test_singlehop_questions_v1.0.json
#     ├── musique_ans_v1.0_dev.jsonl
#     ├── musique_ans_v1.0_test.jsonl
#     ├── musique_ans_v1.0_train.jsonl
#     ├── musique_full_v1.0_dev.jsonl
#     ├── musique_full_v1.0_test.jsonl
#     └── musique_full_v1.0_train.jsonl
