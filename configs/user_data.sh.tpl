#!/bin/bash
# ============================================
# User Data Script for GPU Nodes
# ============================================

set -e

# Set variables
CLUSTER_NAME="${cluster_name}"
NODE_INDEX="${node_index}"
TOTAL_NODES="${total_nodes}"
ECR_REPO="${ecr_repository_url}"

echo "=========================================="
echo "Setting up GPU node ${NODE_INDEX}/${TOTAL_NODES}"
echo "Cluster: ${CLUSTER_NAME}"
echo "=========================================="

# Install NVIDIA drivers
apt-get update
apt-get install -y --no-install-recommends \
    nvidia-driver-535 \
    nvidia-utils-535 \
    cuda-drivers-fabricmanager-535

# Install Docker
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install NCCL prerequisites
apt-get install -y \
    libnccl2 \
    libnccl-dev \
    openmpi-bin \
    openmpi-common

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export NCCL_LOGFILE=/var/log/nccl_trace.log
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Pull Docker image
if [ ! -z "${ECR_REPO}" ]; then
    docker pull ${ECR_REPO}:latest
else
    echo "No ECR repository specified. Using local build..."
fi

# Create directory for NCCL logs
mkdir -p /var/log/nccl

# Run Docker container
docker run --gpus all --network host -d \
    --name ${CLUSTER_NAME}-node-${NODE_INDEX} \
    -e NCCL_DEBUG=INFO \
    -e NCCL_DEBUG_SUBSYS=INIT,COLL,ENV \
    -e NCCL_LOGFILE=/var/log/nccl_trace.log \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e RANK=${NODE_INDEX} \
    -e WORLD_SIZE=${TOTAL_NODES} \
    -e MASTER_ADDR=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4) \
    -v /var/log/nccl:/var/log/nccl \
    ${ECR_REPO}:latest \
    python3 scripts/run_multi_node_benchmark.py

echo "=========================================="
echo "Node ${NODE_INDEX} setup complete!"
echo "=========================================="