"""
Terraform Simulator - offline, deterministic.
"""
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TerraformResource:
    type: str
    name: str
    attributes: Dict[str, any]
    provider: str = "oci"
    id: str = ""


@dataclass
class Cluster:
    name: str
    instances: List[TerraformResource]
    total_nodes: int
    placement_group_name: str
    region: str = "eu-frankfurt-1"

    def to_json(self) -> str:
        return json.dumps({
            "cluster_name": self.name, "region": self.region,
            "placement_group": self.placement_group_name,
            "nodes": [asdict(i) for i in self.instances],
            "total_nodes": self.total_nodes,
        }, indent=2)


@dataclass
class TerraformState:
    resources: List[TerraformResource] = field(default_factory=list)
    outputs: Dict[str, any] = field(default_factory=dict)
    version: int = 4


class TerraformSimulator:
    def __init__(self, region: str = "eu-frankfurt-1", seed: int = 42):
        self.region = region
        self.seed = seed
        self._rng = random.Random(seed)
        self.state = TerraformState()
        self.resources_by_type: Dict[str, TerraformResource] = {}
        self.mocked_resources: Dict[str, Dict] = {}
        self.cluster_name = ""

    def _ip(self) -> str:
        return f"{self._rng.randint(10, 250)}.{self._rng.randint(1, 250)}.{self._rng.randint(1, 250)}.{self._rng.randint(1, 250)}"

    def _id(self, prefix: str) -> str:
        return f"{prefix}-{''.join(self._rng.choices('abcdef0123456789', k=17))}"

    def create_cluster(self, cluster_name: str, node_count: int = 2,
                       instance_type: str = "VM.GPU.A10.1",
                       placement_group: str = "nccl-cluster-pg",
                       mock_mode: bool = True) -> Dict[str, Any]:
        self.cluster_name = cluster_name
        resources, instances = [], []

        pg = TerraformResource(
            type="oci_core_placement_group",
            name=f"{cluster_name}-pg",
            attributes={
                "name": placement_group, "strategy": "cluster",
                "id": self._id("pg"),
            },
        )
        resources.append(pg)
        self.resources_by_type["oci_core_placement_group"] = pg

        for i in range(node_count):
            inst = TerraformResource(
                type="oci_core_instance",
                name=f"{cluster_name}-node-{i}",
                attributes={
                    "shape": instance_type,
                    "image_ocid": "ocid1.image.oc1..mock",
                    "public_ip": self._ip(),
                    "private_ip": self._ip(),
                    "state": "RUNNING",
                    "availability_domain": "Uocm:EU-FRANKFURT-1-AD-1",
                    "tags": {"Name": f"LLM-Systems-Node-{i}", "Cluster": cluster_name},
                    "id": self._id("ocid1.instance"),
                },
            )
            resources.append(inst)
            instances.append(inst)
            self.resources_by_type[f"oci_core_instance_{i}"] = inst

        self.state.resources.extend(resources)
        cluster = Cluster(
            name=cluster_name, instances=instances,
            total_nodes=node_count,
            placement_group_name=placement_group, region=self.region,
        )
        self.state.outputs = {
            "instance_ips": [i.attributes["public_ip"] for i in instances],
            "instance_private_ips": [i.attributes["private_ip"] for i in instances],
            "instance_ids": [i.attributes["id"] for i in instances],
            "placement_group_name": placement_group,
            "cluster_name": cluster_name,
            "total_nodes": node_count,
            "instance_type": instance_type,
            "region": self.region,
        }
        return {
            "cluster_name": cluster_name, "cluster": cluster,
            "resources": resources, "outputs": self.state.outputs,
            "mock_mode": mock_mode,
        }

    def mock_resource(self, resource_type: str, attributes: Dict) -> Dict:
        self.mocked_resources[resource_type] = attributes
        return {"mocked": resource_type, "attributes": attributes}

    def get_resource(self, resource_type: str) -> Optional[TerraformResource]:
        if resource_type in self.mocked_resources:
            return TerraformResource(
                type=resource_type, name="mocked",
                attributes=self.mocked_resources[resource_type],
                id=self._id("mock"),
            )
        return self.resources_by_type.get(resource_type)

    def terraform_test(self, test_config: Dict) -> Dict:
        return {
            "tests": [{"name": f"test_{i}", "passed": True}
                      for i, _ in enumerate(test_config.get("assertions", []))],
            "passed": len(test_config.get("assertions", [])),
            "failed": 0,
        }

    def generate_terraform_plan(self) -> str:
        if not self.cluster_name:
            return "# Error: No cluster defined"
        return (f"# Terraform Plan for {self.cluster_name}\n"
                f"Plan: {self.state.outputs.get('total_nodes', 2)} to add.\n")

    def export_state(self, format: str = "json") -> str:
        state_dict = {
            "version": self.state.version,
            "resources": [asdict(r) for r in self.state.resources],
            "outputs": self.state.outputs,
        }
        return json.dumps(state_dict, indent=2)