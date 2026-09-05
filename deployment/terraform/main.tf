provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "model_repository" {
  name = "llmops-rag-model"
  image_tag_mutability = "MUTABLE"
  
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_lambda_function" "inference_function" {
  function_name = "rag-inference"
  role = var.lambda_role_arn
  package_type = "Image"
  image_uri = "${aws_ecr_repository.model_repository.repository_url}:latest"
  timeout = 300
  memory_size = 4096
  
  environment {
    variables = {
      MODEL_PATH = "/opt/model"
    }
  }
}

resource "aws_api_gateway_rest_api" "rag_api" {
  name = "rag-api"
  description = "RAG Inference API"
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_deployment" "prod" {
  rest_api_id = aws_api_gateway_rest_api.rag_api.id
  stage_name = "prod"
  depends_on = [aws_api_gateway_integration.generate_integration]
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference_function.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_api_gateway_rest_api.rag_api.execution_arn}/*/*"
}