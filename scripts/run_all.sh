#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate

LOG=/tmp/llm-engine-run.log
: > "$LOG"

(cd "$REPO_ROOT/configs" && terraform validate -no-color) >>"$LOG" 2>&1 || true

python "$REPO_ROOT/scripts/run_simulation.py" \
    --nodes 2 --gpus-per-node 4 --iterations 50 \
    --output-dir "$REPO_ROOT/artifacts" --seed 42

python -m pytest "$REPO_ROOT/tests/" -q >>"$LOG" 2>&1 || true

echo ""
echo "Generated artifacts:"
echo "----------------------------------------------------------------------"
ls -1 "$REPO_ROOT/artifacts/" | grep -v "^nccl_traces$" | sed 's/^/  /'
echo "----------------------------------------------------------------------"