#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG=/tmp/llm-engine-setup.log
: > "$LOG"

sudo apt-get update -qq >>"$LOG" 2>&1
sudo apt-get install -y -qq \
    build-essential cmake curl unzip git wget jq \
    libcurl4-openssl-dev \
    python3 python3-pip python3-venv python3-full >>"$LOG" 2>&1

if ! command -v terraform >/dev/null 2>&1; then
    ARCH=$(dpkg --print-architecture)
    cd /tmp
    wget -q "https://releases.hashicorp.com/terraform/1.9.5/terraform_1.9.5_linux_${ARCH}.zip" >>"$LOG" 2>&1
    unzip -q -o "terraform_1.9.5_linux_${ARCH}.zip" >>"$LOG" 2>&1
    sudo mv terraform /usr/local/bin/ >>"$LOG" 2>&1
    sudo chmod +x /usr/local/bin/terraform
    cd "$REPO_ROOT"
fi

python3 -m venv venv >>"$LOG" 2>&1
source venv/bin/activate
pip install --upgrade pip -q >>"$LOG" 2>&1
pip install -q -r requirements.txt >>"$LOG" 2>&1
pip install -q -e . >>"$LOG" 2>&1

(cd "$REPO_ROOT/configs" && terraform init -upgrade -no-color) >>"$LOG" 2>&1 || true

echo "Setup complete."