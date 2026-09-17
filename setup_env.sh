#!/bin/bash
# ============================================
# setup_env.sh — Oracle Cloud Ubuntu (ARM/x86)
# ============================================
set -e

echo "==================================================="
echo "High-Throughput LLM Engine — Environment Setup"
echo "==================================================="

# -------- 1. System packages --------
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake curl unzip git wget jq \
    libcurl4-openssl-dev \
    python3 python3-pip python3-venv python3-full

# -------- 2. Terraform (optional) --------
echo "[2/5] Installing Terraform (if not present)..."
if ! command -v terraform >/dev/null 2>&1; then
    bash scripts/install_terraform.sh || true
else
    echo " Terraform already installed: $(terraform version -json | jq -r .terraform_version)"
fi

# -------- 3. Python venv --------
echo "[3/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip

# -------- 4. Python deps --------
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt
pip install -e .

# -------- 5. Artifacts dir --------
echo "[5/5] Creating directories..."
mkdir -p artifacts/nccl_traces
mkdir -p benchmarks/build

echo "==================================================="
echo " Setup complete!"
echo "==================================================="
echo "Activate venv: source venv/bin/activate"
echo "Run offline verification:  make verify-offline"
echo "Run simulation:            make simulation"
echo "Validate Terraform:        make terraform-validate"
echo "==================================================="