"""
Deployment module for LLMOps RAG Pipeline

This module contains deployment scripts and configurations for:
- AWS ECR/Lambda/API Gateway deployment
- FastAPI inference service
- Terraform infrastructure as code
"""

__version__ = "1.0.0"
__author__ = "MisterFOUR"

from deployment.inference_api import app, InferenceRequest, InferenceResponse
from deployment.aws_deploy import AWSDeployer, deploy_to_aws

__all__ = [
    "app",
    "InferenceRequest", 
    "InferenceResponse",
    "AWSDeployer",
    "deploy_to_aws"
]