#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate

LOG=/tmp/llm-engine-run.log
: > "$LOG"

# Optional Terraform validation (silent, non-fatal)
(cd "$REPO_ROOT/configs" && terraform validate -no-color) >>"$LOG" 2>&1 || true

# Run simulation (visible output)
python "$REPO_ROOT/scripts/run_simulation.py" \
    --nodes 2 --gpus-per-node 4 --iterations 50 \
    --output-dir "$REPO_ROOT/artifacts" --seed 42

# Run tests (silent)
python -m pytest "$REPO_ROOT/tests/" -q >>"$LOG" 2>&1 || true

# Display summary
echo ""
python "$REPO_ROOT/scripts/display_results.py" "$REPO_ROOT/artifacts/simulation_report.json"

# List artifacts
echo ""
echo "Generated artifacts:"
echo "----------------------------------------------------------------------"
ls -1 "$REPO_ROOT/artifacts/" | grep -v "^nccl_traces$" | sed 's/^/  /'
echo "----------------------------------------------------------------------"