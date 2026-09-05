# ============================================
# configs/variables.tf
# ============================================

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-central-1"
}

variable "cluster_name" {
  description = "Name of the cluster"
  type        = string
  default     = "nccl-cluster"
}

variable "node_count" {
  description = "Number of GPU nodes to provision"
  type        = number
  default     = 2
}

variable "instance_type" {
  description = "EC2 instance type for GPU nodes"
  type        = string
  default     = "g5.12xlarge"
}

variable "ami_id" {
  description = "AMI ID for the GPU instances (Deep Learning AMI recommended)"
  type        = string
  default     = "ami-0abcdef1234567890"
}

variable "ssh_key_name" {
  description = "Name of the SSH key pair to use for EC2 instances"
  type        = string
}

variable "ecr_repository_url" {
  description = "URL of the ECR repository containing the multi-node Docker image"
  type        = string
  default     = ""
}

variable "vpc_id" {
  description = "VPC ID for the security group"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags for resources"
  type        = map(string)
  default = {
    Project     = "High-Throughput-LLM-Engine"
    Environment = "development"
    ManagedBy   = "Terraform"
  }
}