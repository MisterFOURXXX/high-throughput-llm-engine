import boto3
import json
import subprocess
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AWSDeployer:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.ecr_client = boto3.client('ecr', region_name=region_name)
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.api_client = boto3.client('apigateway', region_name=region_name)
        self.ecs_client = boto3.client('ecs', region_name=region_name)
    
    def create_ecr_repository(self, repository_name: str):
        try:
            response = self.ecr_client.create_repository(
                repositoryName=repository_name,
                imageTagMutability="MUTABLE",
                encryptionConfiguration={"encryptionType": "AES256"}
            )
            repository_uri = response['repository']['repositoryUri']
            print(f"ECR repository created: {repository_uri}")
            return repository_uri
        except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
            response = self.ecr_client.describe_repositories(repositoryNames=[repository_name])
            repository_uri = response['repositories'][0]['repositoryUri']
            print(f"ECR repository already exists: {repository_uri}")
            return repository_uri
    
    def build_and_push_image(self, repository_uri: str, image_tag: str = "latest"):
        # Login to ECR
        login_result = subprocess.run(
            f"aws ecr get-login-password --region {self.region_name} | docker login --username AWS --password-stdin {repository_uri.split('/')[0]}",
            shell=True,
            capture_output=True,
            text=True
        )
        if login_result.returncode != 0:
            print(f"Docker login failed: {login_result.stderr}")
            return False
        
        # Build Docker image
        build_command = f"docker build -t {repository_uri}:{image_tag} ."
        build_result = subprocess.run(build_command, shell=True, capture_output=True, text=True)
        if build_result.returncode != 0:
            print(f"Docker build failed: {build_result.stderr}")
            return False
        
        # Push image
        push_command = f"docker push {repository_uri}:{image_tag}"
        push_result = subprocess.run(push_command, shell=True, capture_output=True, text=True)
        if push_result.returncode != 0:
            print(f"Docker push failed: {push_result.stderr}")
            return False
        
        print(f"Image pushed to {repository_uri}:{image_tag}")
        return True
    
    def create_lambda_function(self, function_name: str, image_uri: str, role_arn: str):
        try:
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                PackageType='Image',
                Code={'ImageUri': image_uri},
                Role=role_arn,
                Timeout=300,
                MemorySize=4096,
                Environment={'Variables': {'MODEL_PATH': '/opt/model'}}
            )
            print(f"Lambda function created: {function_name}")
            return response['FunctionArn']
        except self.lambda_client.exceptions.ResourceConflictException:
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ImageUri=image_uri,
                Publish=True
            )
            print(f"Lambda function updated: {function_name}")
            return response['FunctionArn']
        except Exception as e:
            print(f"Error creating Lambda function: {e}")
            return None
    
    def create_api_gateway(self, api_name: str, lambda_arn: str):
        try:
            api_response = self.api_client.create_rest_api(
                name=api_name,
                description='RAG Inference API',
                endpointConfiguration={'types': ['REGIONAL']}
            )
            api_id = api_response['id']
            
            root_resource_id = self.api_client.get_resources(restApiId=api_id)['items'][0]['id']
            
            resource_response = self.api_client.create_resource(
                restApiId=api_id,
                parentId=root_resource_id,
                pathPart='generate'
            )
            resource_id = resource_response['id']
            
            self.api_client.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                authorizationType='NONE'
            )
            
            self.api_client.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f'arn:aws:apigateway:{self.region_name}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
            )
            
            # Add CORS support
            self.api_client.put_method(
                restApiId=api_id,
                resourceId=root_resource_id,
                httpMethod='OPTIONS',
                authorizationType='NONE'
            )
            
            self.api_client.put_integration(
                restApiId=api_id,
                resourceId=root_resource_id,
                httpMethod='OPTIONS',
                type='MOCK',
                requestTemplates={'application/json': '{"statusCode": 200}'}
            )
            
            self.api_client.create_deployment(
                restApiId=api_id,
                stageName='prod'
            )
            
            api_url = f"https://{api_id}.execute-api.{self.region_name}.amazonaws.com/prod/generate"
            print(f"API Gateway created: {api_url}")
            return api_url
            
        except Exception as e:
            print(f"Error creating API Gateway: {e}")
            return None

def deploy_to_aws(config: Dict[str, Any]):
    deployer = AWSDeployer(region_name=config['deployment']['aws_region'])
    
    repository_uri = deployer.create_ecr_repository(config['deployment']['ecr_repository_name'])
    if not repository_uri:
        print("Failed to create ECR repository")
        return None
    
    success = deployer.build_and_push_image(repository_uri, "latest")
    if not success:
        print("Failed to build and push image")
        return None
    
    lambda_role_arn = os.getenv('LAMBDA_ROLE_ARN')
    if not lambda_role_arn:
        print("LAMBDA_ROLE_ARN environment variable not set")
        return None
    
    lambda_arn = deployer.create_lambda_function(
        config['deployment']['lambda_function_name'],
        f"{repository_uri}:latest",
        lambda_role_arn
    )
    if not lambda_arn:
        print("Failed to create Lambda function")
        return None
    
    api_url = deployer.create_api_gateway(
        config['deployment']['api_gateway_name'],
        lambda_arn
    )
    
    return {
        'repository_uri': repository_uri,
        'lambda_arn': lambda_arn,
        'api_url': api_url
    }

if __name__ == "__main__":
    import yaml
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    result = deploy_to_aws(config)
    
    if result:
        print("\n" + "="*60)
        print("AWS DEPLOYMENT COMPLETE")
        print("="*60)
        print(f"ECR Repository: {result['repository_uri']}")
        print(f"Lambda ARN: {result['lambda_arn']}")
        print(f"API URL: {result['api_url']}")
        print("="*60)
    else:
        print("\nAWS DEPLOYMENT FAILED")