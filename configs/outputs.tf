output "instance_ips" {
  value = oci_core_instance.gpu_node[*].public_ip
}

output "instance_private_ips" {
  value = oci_core_instance.gpu_node[*].private_ip
}

output "instance_ids" {
  value = oci_core_instance.gpu_node[*].id
}