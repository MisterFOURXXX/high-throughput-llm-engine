variable "tenancy_ocid" {
  description = "OCI Tenancy OCID"
  type        = string
  default     = "ocid1.tenancy.oc1..placeholder"
}

variable "user_ocid" {
  description = "OCI User OCID"
  type        = string
  default     = "ocid1.user.oc1..placeholder"
}

variable "fingerprint" {
  description = "OCI API key fingerprint"
  type        = string
  default     = "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
}

variable "private_key_path" {
  description = "Path to OCI private key"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "region" {
  description = "OCI region"
  type        = string
  default     = "eu-frankfurt-1"
}

variable "compartment_ocid" {
  description = "Compartment OCID"
  type        = string
  default     = "ocid1.compartment.oc1..placeholder"
}

variable "cluster_name" {
  type    = string
  default = "nccl-cluster"
}

variable "node_count" {
  type    = number
  default = 2
}

variable "instance_shape" {
  type    = string
  default = "VM.GPU.A10.1"
}

variable "instance_ocpus" {
  type    = number
  default = 4
}

variable "instance_memory_gb" {
  type    = number
  default = 64
}

variable "instance_image_ocid" {
  type    = string
  default = "ocid1.image.oc1..placeholder"
}

variable "boot_volume_size_gb" {
  type    = number
  default = 200
}

variable "create_network" {
  type    = bool
  default = true
}

variable "subnet_id" {
  type    = string
  default = ""
}

variable "ssh_public_key" {
  type    = string
  default = "ssh-rsa AAAAB3NzaC1yc2E... placeholder"
}

variable "container_image_url" {
  type    = string
  default = ""
}