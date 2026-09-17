# ============================================
# configs/outputs.tf
# ============================================

output "instance_ips" {
  description = "Public IPs of the nodes"
  value       = oci_core_instance.gpu_node[*].public_ip
}

output "instance_private_ips" {
  description = "Private IPs of the nodes"
  value       = oci_core_instance.gpu_node[*].private_ip
}

output "instance_ids" {
  description = "OCIDs of the nodes"
  value       = oci_core_instance.gpu_node[*].id
}

output "instance_names" {
  description = "Display names of the nodes"
  value       = oci_core_instance.gpu_node[*].display_name
}

output "cluster_name" {
  value = var.cluster_name
}

output "region" {
  value = var.region
}

output "vcn_id" {
  value = var.create_network ? oci_core_vcn.nccl_vcn[0].id : null
}

output "subnet_id" {
  value = var.create_network ? oci_core_subnet.nccl_subnet[0].id : var.subnet_id
}