#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/clean-local.sh [--apply] [--artifacts] [--heavy]

Dry-runs local cleanup by default.

Options:
  --apply       Delete the listed local files/directories.
  --artifacts   Include generated results/artifact directories.
  --heavy       Include expensive-to-recreate local directories: .venv, .worktrees, vendor.
USAGE
}

apply=0
include_artifacts=0
include_heavy=0

while (($#)); do
  case "$1" in
    --apply) apply=1 ;;
    --artifacts) include_artifacts=1 ;;
    --heavy) include_heavy=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

declare -a paths=(
  ".pytest_cache"
  "__pycache__"
  "modules/__pycache__"
  "tests/__pycache__"
  "tests/integration/__pycache__"
  "tools/__pycache__"
)

while IFS= read -r path; do
  paths+=("$path")
done < <(
  find . \
    \( -path './.git' -o -path './.venv' -o -path './.worktrees' -o -path './vendor' \) -prune \
    -o -type d -name __pycache__ -print \
    | sed 's#^\./##' \
    | sort -u
)

if ((include_artifacts)); then
  while IFS= read -r path; do
    paths+=("$path")
  done < <(
    find results -mindepth 1 -maxdepth 1 -type d \
      \( -name 'fee-*' -o -name 'hive-hints-truth-*' -o -name 'module-loop-*' -o -name 'rebalancer-polar-mcp-*' \) \
      -print 2>/dev/null | sort -u
  )
  paths+=("artifacts")
fi

if ((include_heavy)); then
  paths+=(".venv" ".worktrees" "vendor")
fi

seen=""
for path in "${paths[@]}"; do
  [[ -e "$path" ]] || continue
  case "
$seen
" in
    *"
$path
"*) continue ;;
  esac
  seen="${seen}${path}"$'\n'

  if ((apply)); then
    rm -rf -- "$path"
    echo "removed $path"
  else
    echo "would remove $path"
  fi
done

if ((!apply)); then
  echo
  echo "Dry run only. Re-run with --apply to delete listed paths."
fi
