#!/bin/bash
# Full repository reset
set -e
cd "$(dirname "$0")/.."
echo "=== Resetting repo at $(pwd) ==="

NUKE=false
CLEAN_TF=false
for arg in "$@"; do
    case $arg in
        --nuke)      NUKE=true ;;
        --terraform) CLEAN_TF=true ;;
    esac
done

deactivate 2>/dev/null || true
rm -rf venv/ .venv/ env/ ENV/ 2>/dev/null || sudo rm -rf venv/ .venv/

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
find . -type d \( -name "*.egg-info" -o -name ".pytest_cache" \
    -o -name ".mypy_cache" -o -name ".ruff_cache" \
    -o -name ".ipynb_checkpoints" \) -exec rm -rf {} + 2>/dev/null || true
rm -rf build/ dist/ .eggs/ htmlcov/ .coverage .coverage.* 2>/dev/null || true

pip cache purge 2>/dev/null || true
rm -rf ~/.cache/torch/ ~/.triton/ ~/.cache/triton/ \
       ~/.cache/flashinfer/ ~/.nv/ 2>/dev/null || true

rm -rf benchmarks/build/ artifacts/*
touch artifacts/.gitkeep
rm -f benchmark_results.csv dynmoe_evaluation_results.csv
rm -rf wandb/ mlruns/ mlartifacts/ .dagshub/ 2>/dev/null || true

if [ "$CLEAN_TF" = true ]; then
    cd configs && terraform destroy -auto-approve 2>/dev/null || true; cd ..
fi
rm -rf configs/.terraform configs/.terraform.lock.hcl \
       configs/terraform.tfstate configs/terraform.tfstate.backup \
       configs/tfplan 2>/dev/null || true

if [ "$NUKE" = true ]; then
    rm -f configs/terraform.tfvars configs/*.auto.tfvars 2>/dev/null || true
fi

echo "✓ Reset complete"