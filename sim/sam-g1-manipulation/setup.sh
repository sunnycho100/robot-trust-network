#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bigym_dir="$task_dir/../../../bigym2-lab"
bigym_commit=d71501359b215b9d78e27dcb70b220412d474dfa
if ! command -v uv >/dev/null; then
  echo "Install uv from https://docs.astral.sh/uv/ then rerun this script." >&2
  exit 1
fi
if [[ ! -e "$bigym_dir" ]]; then
  git clone --no-checkout https://github.com/swirl-uk/BiGym2.git "$bigym_dir"
  git -C "$bigym_dir" checkout --detach "$bigym_commit"
fi
actual_commit="$(git -C "$bigym_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$bigym_commit" ]]; then
  echo "Expected BiGym commit $bigym_commit; found $actual_commit." >&2
  echo "Resolve the dependency checkout explicitly; setup will not reset it." >&2
  exit 1
fi
cd "$task_dir"
uv sync --frozen --python 3.12
echo "Ready. Run: uv run --no-sync python run.py smoke"
echo "Mac GUI: uv run --no-sync python view.py console"
