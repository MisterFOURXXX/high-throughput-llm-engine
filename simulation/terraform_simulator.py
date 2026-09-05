"""
Terraform Simulator
Simulates Terraform infrastructure provisioning with mock support.
"""

import json
import yaml
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TerraformResource:
    """Simulated Terraform resource."""
    type: str
    name: str
    attributes: Dict[str, any]
    provider: str = "aws"
    id: str = ""


@dataclass
class TerraformState:
    """Simulated Terraform state."""
    resources: List[TerraformResource] = field(default_factory=list)
    outputs: Dict[str, any] = field(default_factory=dict)
    version: int = 4


class TerraformSimulator:
    """
    Simulates Terraform provisioning with support for mock testing.
    """
    
    def __init__(self, region: str = "eu-central-1"):
        self.region = region
        self.state = TerraformState()
        self.resources_by_type = {}
        self.mocked_resources = {}
        self.cluster_name = ""
    
    def generate_ip(self) -> str:
        """Generate a random IP address."""
        return f"{random.randint(10, 255)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def generate_id(self, prefix: str) -> str:
        """Generate a random resource ID."""
        return f"{prefix}-{''.join(random.choices('abcdef0123456789', k=17))}"
    
    def create_cluster(
        self,
        cluster_name: str,
        node_count: int = 2,
        instance_type: str = "g5.12xlarge",
        placement_group: str = "llm-serving-cluster-pg",
        mock_mode: bool = True
    ) -> Dict[str, Any]:
        """Simulate creating a cluster."""
        self.cluster_name = cluster_name
        resources = []
        instances = []
        
        # 1. Placement Group
        pg_resource = TerraformResource(
            type="aws_placement_group",
            name=f"{cluster_name}-pg",
            attributes={
                "name": placement_group,
                "strategy": "cluster",
                "id": self.generate_id("pg"),
            }
        )
        resources.append(pg_resource)
        self.resources_by_type["aws_placement_group"] = pg_resource
        
        # 2. EC2 Instances
        for i in range(node_count):
            instance = TerraformResource(
                type="aws_instance",
                name=f"{cluster_name}-node-{i}",
                attributes={
                    "ami": "ami-0abcdef1234567890" if mock_mode else "<real-ami-id>",
                    "instance_type": instance_type,
                    "placement_group": placement_group,
                    "public_ip": self.generate_ip() if mock_mode else "<real-ip>",
                    "private_ip": self.generate_ip() if mock_mode else "<real-ip>",
                    "state": "running",
                    "tags": {
                        "Name": f"LLM-Systems-Node-{i}",
                        "Cluster": cluster_name,
                    },
                    "id": self.generate_id("i"),
                }
            )
            resources.append(instance)
            instances.append(instance)
            self.resources_by_type[f"aws_instance_{i}"] = instance
        
        self.state.resources.extend(resources)
        
        self.state.outputs = {
            "instance_ips": [i.attributes["public_ip"] for i in instances],
            "instance_ids": [i.attributes["id"] for i in instances],
            "placement_group_name": placement_group,
            "cluster_name": cluster_name,
            "total_nodes": node_count,
            "instance_type": instance_type,
            "region": self.region,
        }
        
        return {
            "cluster_name": cluster_name,
            "resources": resources,
            "outputs": self.state.outputs,
            "mock_mode": mock_mode,
        }
    
    def mock_resource(self, resource_type: str, attributes: Dict) -> None:
        """Simulate Terraform's mock_resource block."""
        self.mocked_resources[resource_type] = attributes
        return {"mocked": resource_type, "attributes": attributes}
    
    def get_resource(self, resource_type: str) -> Optional[TerraformResource]:
        """Get a resource (real or mocked)."""
        if resource_type in self.mocked_resources:
            return TerraformResource(
                type=resource_type,
                name="mocked",
                attributes=self.mocked_resources[resource_type],
                id=self.generate_id("mock")
            )
        return self.resources_by_type.get(resource_type)
    
    def terraform_test(self, test_config: Dict) -> Dict:
        """Simulate running terraform test with mocks."""
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0,
            "duration_sec": random.uniform(5, 30),
        }
        
        for i, assertion in enumerate(test_config.get("assertions", [])):
            test_result = {
                "name": f"test_{i}",
                "condition": assertion.get("condition", ""),
                "passed": random.random() > 0.1,
                "error_message": assertion.get("error_message", ""),
            }
            results["tests"].append(test_result)
            if test_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def generate_terraform_plan(self) -> str:
        """Generate a simulated terraform plan output."""
        if not self.cluster_name:
            return "# Error: No cluster defined"
        
        return f"""
# Terraform Plan for {self.cluster_name}
# Generated: {datetime.now().isoformat()}

Resource changes:
  + aws_placement_group.nccl_cluster
      id:   <computed>
      name: "llm-serving-cluster-pg"
      strategy: "cluster"

  + aws_instance.gpu_node[{self.state.outputs.get('total_nodes', 2)}]
      id:   <computed>
      ami:  "ami-0abcdef1234567890"
      instance_type: "{self.state.outputs.get('instance_type', 'g5.12xlarge')}"
      placement_group: aws_placement_group.nccl_cluster.id
      
      + ebs_block_device {{
          delete_on_termination: true
          device_name: "/dev/sda1"
          volume_size: 200
          volume_type: "gp3"
        }}
      
      tags = {{
        Name = "LLM-Systems-Node-*"
      }}

Plan: {self.state.outputs.get('total_nodes', 2)} to add, 0 to change, 0 to destroy.
"""
    
    def export_state(self, format: str = "json") -> str:
        """Export Terraform state."""
        state_dict = {
            "version": self.state.version,
            "terraform_version": "1.5.0",
            "resources": [asdict(r) for r in self.state.resources],
            "outputs": self.state.outputs,
        }
        
        if format == "json":
            return json.dumps(state_dict, indent=2)
        elif format == "yaml":
            return yaml.dump(state_dict)
        else:
            return str(state_dict)