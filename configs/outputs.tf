output "instance_ips" {
  description = "Public IP addresses of the GPU nodes"
  value       = aws_instance.gpu_node[*].public_ip
}

output "instance_ids" {
  description = "Instance IDs of the GPU nodes"
  value       = aws_instance.gpu_node[*].id
}

output "placement_group_name" {
  description = "Name of the placement group"
  value       = aws_placement_group.nccl_cluster.name
}