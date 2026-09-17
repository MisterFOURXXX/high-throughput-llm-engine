#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

deactivate 2>/dev/null || true
rm -rf venv/ .venv/ env/ ENV/ 2>/dev/null || sudo rm -rf venv/ .venv/ env/ ENV/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
find . -type d \( -name "*.egg-info" -o -name ".pytest_cache" \
    -o -name ".mypy_cache" -o -name ".ruff_cache" \) -exec rm -rf {} + 2>/dev/null || true
rm -rf build/ dist/ .eggs/ htmlcov/ .coverage .coverage.* 2>/dev/null || true
rm -rf benchmarks/build/ 2>/dev/null || true
rm -rf artifacts/* 2>/dev/null || true
mkdir -p artifacts/nccl_traces
touch artifacts/.gitkeep
rm -rf configs/.terraform configs/.terraform.lock.hcl \
       configs/terraform.tfstate configs/terraform.tfstate.backup \
       configs/tfplan 2>/dev/null || true
echo "Cleanup complete."