#!/bin/bash
# ============================================
# run_all_experiments.sh
# Full lifecycle: reset → setup → validate → simulate
# ============================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo " High-Throughput LLM Engine — Full Experiment Runner"
echo " Repo root: $REPO_ROOT"
echo "=========================================================="

# ---------- Step 0: Ensure we are at repo root ----------
if [ ! -f "Makefile" ] || [ ! -d "simulation" ]; then
    echo "✗ Makefile or simulation/ missing. Aborting."
    exit 1
fi

# ---------- Step 1: Full reset ----------
echo ""
echo "[1/6] Full reset (remove venv, artifacts, Terraform state)..."
rm -rf venv
rm -rf artifacts/*
rm -f  benchmark_results.csv
rm -rf benchmarks/build
rm -rf configs/.terraform \
       configs/.terraform.lock.hcl \
       configs/terraform.tfstate \
       configs/terraform.tfstate.backup \
       configs/tfplan
mkdir -p artifacts/nccl_traces
echo "✓ Reset done"

# ---------- Step 2: System setup ----------
echo ""
echo "[2/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake curl unzip git wget jq \
    libcurl4-openssl-dev \
    python3 python3-pip python3-venv python3-full

# ---------- Step 3: Terraform ----------
echo ""
echo "[3/6] Ensuring Terraform is installed..."
if ! command -v terraform >/dev/null 2>&1; then
    bash scripts/install_terraform.sh
fi
terraform version

# ---------- Step 4: Python env ----------
echo ""
echo "[4/6] Creating Python virtual environment..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# ---------- Step 5: Terraform validate (offline) ----------
echo ""
echo "[5/6] Validating Terraform configuration (offline)..."
cd configs
terraform init -upgrade
terraform validate
cd ..

# ---------- Step 6: Run simulation ----------
echo ""
echo "[6/6] Running NCCL simulation (offline)..."
python scripts/run_simulation.py \
    --nodes 2 --gpus-per-node 4 --iterations 50 \
    --output-dir artifacts

# ---------- Summary ----------
echo ""
echo "=========================================================="
echo " ✓ EXPERIMENT COMPLETE"
echo "=========================================================="
echo "Artifacts:"
ls -la artifacts/
echo ""
echo "To activate venv in future shells: source venv/bin/activate"
echo "To re-run simulation:              make simulation"
echo "To deploy to real OCI:             make terraform-apply"
echo "=========================================================="