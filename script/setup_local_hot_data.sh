#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOT_ROOT="${RESCORE_HOT_ROOT:-/docker/data/${USER}/ReSCORE}"
DRY_RUN="${RESCORE_HOT_DATA_DRY_RUN:-0}"

note() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1"
}

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

resolve_path_for_reporting() {
  local path="$1"
  if [[ -e "$path" ]]; then
    realpath "$path"
  else
    printf '%s\n' "$path"
  fi
}

migrate_path() {
  local relative_path="$1"
  local source_path="$REPO_ROOT/$relative_path"
  local target_path="$HOT_ROOT/$relative_path"

  run mkdir -p "$(dirname "$target_path")"

  if [[ -L "$source_path" ]]; then
    note "Skipping ${relative_path}; already symlinked to $(readlink "$source_path")"
    return 0
  fi

  if [[ ! -e "$source_path" ]]; then
    run mkdir -p "$target_path"
    run mkdir -p "$(dirname "$source_path")"
    run ln -s "$target_path" "$source_path"
    note "Created empty hot-data symlink ${relative_path} -> ${target_path}"
    return 0
  fi

  if [[ -e "$target_path" ]]; then
    note "Refusing to migrate ${relative_path}; target already exists at ${target_path}"
    note "Resolve the existing target manually, then rerun this script."
    return 1
  fi

  run mv "$source_path" "$target_path"
  run ln -s "$target_path" "$source_path"
  note "Migrated ${relative_path} -> ${target_path} and linked it back into the repo."
}

note "Repo root: ${REPO_ROOT}"
note "Hot-data root: ${HOT_ROOT}"
note "Mode: $( [[ "$DRY_RUN" == "1" ]] && printf 'dry-run' || printf 'apply' )"

run mkdir -p "$HOT_ROOT"

migrate_path "logs"
migrate_path "predictions"
migrate_path "data/database"

note "Hot-data migration complete."
note "Real paths after migration:"
note "  logs -> $(resolve_path_for_reporting "$REPO_ROOT/logs")"
note "  predictions -> $(resolve_path_for_reporting "$REPO_ROOT/predictions")"
note "  data/database -> $(resolve_path_for_reporting "$REPO_ROOT/data/database")"
note "You can now rerun build/train; runtime logs, checkpoints, pipeline JSON logs, and SQLite docstore writes will go to /docker."
