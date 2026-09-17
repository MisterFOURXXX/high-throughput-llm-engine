#!/bin/bash
# ============================================================
# Oracle Cloud-init for GPU / CPU Nodes
# All bash variables that must survive Terraform use $${VAR}
# ============================================================
set -e

export DEBIAN_FRONTEND=noninteractive

CLUSTER_NAME="${cluster_name}"
NODE_INDEX="${node_index}"
TOTAL_NODES="${total_nodes}"
CONTAINER_IMAGE="${container_image_url}"

echo "Initializing OCI node $${NODE_INDEX}/$${TOTAL_NODES}"
echo "Cluster: $${CLUSTER_NAME}"

apt-get update
apt-get install -y --no-install-recommends \
    curl wget git unzip ca-certificates \
    build-essential python3 python3-pip python3-venv docker.io

systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu || true

# NVIDIA toolkit (only if GPU present)
if lspci 2>/dev/null | grep -qi nvidia; then
    distribution="$(. /etc/os-release; echo $${ID}$${VERSION_ID})"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit || true
fi

cat > /etc/environment <<'ENVEOF'
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
NCCL_LOGFILE=/var/log/nccl_trace.log
NCCL_SOCKET_IFNAME=ens3
NCCL_IB_DISABLE=1
ENVEOF

mkdir -p /var/log/nccl

if [ -n "$${CONTAINER_IMAGE}" ]; then
    docker pull "$${CONTAINER_IMAGE}" || true
fi

echo "Node $${NODE_INDEX} setup complete."