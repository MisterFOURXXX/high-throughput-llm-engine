#!/bin/bash
# ============================================================
# setup_and_run.sh — one-shot setup + run
# ============================================================
set -e

cd "$(dirname "$0")/.."
echo "=== Working in: $(pwd) ==="

echo "[1/6] Cleaning previous state..."
deactivate 2>/dev/null || true
rm -rf venv artifacts/* benchmarks/build 2>/dev/null || true
rm -rf configs/.terraform configs/.terraform.lock.hcl \
       configs/terraform.tfstate* configs/tfplan 2>/dev/null || true
mkdir -p artifacts/nccl_traces
touch artifacts/.gitkeep

echo "[2/6] Ensuring Terraform is installed..."
if ! command -v terraform >/dev/null 2>&1; then
    bash scripts/install_terraform.sh
else
    echo " Terraform: $(terraform version | head -1)"
fi

echo "[3/6] Creating Python venv..."
python3 -m venv venv
source venv/bin/activate

echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "[5/6] Validating Terraform..."
cd configs && terraform init -upgrade >/dev/null && terraform validate && cd ..

echo "[6/6] Running simulation..."
python scripts/run_simulation.py \
    --nodes 2 --gpus-per-node 4 --iterations 50 \
    --output-dir artifacts

echo ""
echo "=========================================================="
echo "✓ DONE! Artifacts:"
ls -1 artifacts/
echo "=========================================================="