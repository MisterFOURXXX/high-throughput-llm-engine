#!/bin/bash
# Environment setup for High-Throughput LLM Engine

set -e  # Exit on error

echo "==================================================="
echo "Setting up High-Throughput LLM Engine environment"
echo "==================================================="

# ------------------------------------------------------------------
# 1. Install system dependencies (Ubuntu/Debian)
# ------------------------------------------------------------------
echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    python3-pip \
    python3-venv \
    git \
    wget

# ------------------------------------------------------------------
# 2. Setup Python virtual environment (optional)
# ------------------------------------------------------------------
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created at ./venv"
fi
source venv/bin/activate

# ------------------------------------------------------------------
# 3. Install Python dependencies
# ------------------------------------------------------------------
echo "[3/5] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------------
# 4. Build the C++ load generator
# ------------------------------------------------------------------
echo "[4/5] Building C++ load generator..."
make build-benchmark

# ------------------------------------------------------------------
# 5. Create necessary directories
# ------------------------------------------------------------------
echo "[5/5] Creating directories..."
mkdir -p artifacts/nccl_traces
mkdir -p benchmarks/build

echo "==================================================="
echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run a benchmark: make run-benchmark"
echo "==================================================="