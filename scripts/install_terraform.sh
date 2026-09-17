#!/bin/bash
# Install Terraform on Oracle Cloud Ubuntu VM (ARM/x86)
set -e
TERRAFORM_VERSION="${TERRAFORM_VERSION:-1.9.5}"
ARCH=$(dpkg --print-architecture)

echo "Installing Terraform ${TERRAFORM_VERSION} for ${ARCH}..."
sudo apt-get install -y unzip wget >/dev/null 2>&1 || true
cd /tmp
wget -q "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${ARCH}.zip"
unzip -o "terraform_${TERRAFORM_VERSION}_linux_${ARCH}.zip"
sudo mv -f terraform /usr/local/bin/
sudo chmod +x /usr/local/bin/terraform
rm -f "terraform_${TERRAFORM_VERSION}_linux_${ARCH}.zip"
echo "✓ $(terraform version | head -1)"