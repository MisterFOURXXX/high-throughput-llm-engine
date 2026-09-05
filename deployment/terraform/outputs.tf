output "ecr_repository_url" {
  description = "URL of ECR repository"
  value       = aws_ecr_repository.model_repository.repository_url
}

output "lambda_function_arn" {
  description = "ARN of Lambda function"
  value       = aws_lambda_function.inference_function.arn
}

output "api_gateway_url" {
  description = "URL of API Gateway endpoint"
  value       = "${aws_api_gateway_deployment.prod.invoke_url}/generate"
}

output "api_gateway_id" {
  description = "ID of API Gateway"
  value       = aws_api_gateway_rest_api.rag_api.id
}

output "lambda_function_name" {
  description = "Name of Lambda function"
  value       = aws_lambda_function.inference_function.function_name
}