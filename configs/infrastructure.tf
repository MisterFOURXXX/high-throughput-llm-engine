# ============================================
# configs/infrastructure.tf
# Multi-node AWS Infrastructure with Mock Support
# ============================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Random suffix for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# Placement Group for low-latency network
resource "aws_placement_group" "nccl_cluster" {
  name     = "${var.cluster_name}-pg-${random_id.suffix.hex}"
  strategy = "cluster"
}

# Security Group for NCCL communication
resource "aws_security_group" "nccl_sg" {
  name        = "${var.cluster_name}-sg-${random_id.suffix.hex}"
  description = "Security group for NCCL multi-node communication"
  vpc_id      = var.vpc_id != "" ? var.vpc_id : null

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Allow all internal traffic between nodes"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "vLLM API server"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# EC2 Instances with GPU
resource "aws_instance" "gpu_node" {
  count = var.node_count

  ami                  = var.ami_id
  instance_type        = var.instance_type
  placement_group      = aws_placement_group.nccl_cluster.id
  key_name             = var.ssh_key_name
  vpc_security_group_ids = [aws_security_group.nccl_sg.id]

  user_data = templatefile("${path.module}/user_data.sh", {
    ecr_repository_url = var.ecr_repository_url
    cluster_name       = var.cluster_name
    node_index         = count.index
    total_nodes        = var.node_count
  })

  root_block_device {
    volume_size = 200
    volume_type = "gp3"
    delete_on_termination = true
  }

  tags = merge(var.tags, {
    Name = "${var.cluster_name}-node-${count.index}"
    Cluster = var.cluster_name
    Role = "gpu-worker"
    NodeIndex = count.index
  })
}

# Outputs
output "instance_ips" {
  description = "Public IP addresses of the GPU nodes"
  value       = aws_instance.gpu_node[*].public_ip
}

output "instance_ids" {
  description = "Instance IDs of the GPU nodes"
  value       = aws_instance.gpu_node[*].id
}

output "instance_private_ips" {
  description = "Private IP addresses of the GPU nodes"
  value       = aws_instance.gpu_node[*].private_ip
}

output "placement_group_name" {
  description = "Name of the placement group"
  value       = aws_placement_group.nccl_cluster.name
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.nccl_sg.id
}