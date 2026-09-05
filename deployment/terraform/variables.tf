variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "lambda_role_arn" {
  description = "ARN of the IAM role for Lambda function"
  type        = string
}

variable "ecr_repository_name" {
  description = "Name of ECR repository"
  type        = string
  default     = "llmops-rag-model"
}

variable "lambda_function_name" {
  description = "Name of Lambda function"
  type        = string
  default     = "rag-inference"
}

variable "api_gateway_name" {
  description = "Name of API Gateway"
  type        = string
  default     = "rag-api"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "tags" {
  description = "Tags for resources"
  type        = map(string)
  default = {
    Project     = "LLMOps-RAG-Pipeline"
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}