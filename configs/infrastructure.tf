# ============================================================
# configs/infrastructure.tf
# OCI Multi-Node Infrastructure
# ============================================================
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    oci    = { source = "oracle/oci",       version = "~> 6.0" }
    random = { source = "hashicorp/random", version = "~> 3.0" }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

resource "random_id" "suffix" { byte_length = 4 }

resource "oci_core_vcn" "nccl_vcn" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_ocid
  cidr_block     = "10.0.0.0/16"
  display_name   = "${var.cluster_name}-vcn-${random_id.suffix.hex}"
  dns_label      = "ncclvcn"
}

resource "oci_core_internet_gateway" "nccl_igw" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.nccl_vcn[0].id
  display_name   = "${var.cluster_name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "nccl_rt" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.nccl_vcn[0].id
  display_name   = "${var.cluster_name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.nccl_igw[0].id
  }
}

resource "oci_core_security_list" "nccl_sl" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.nccl_vcn[0].id
  display_name   = "${var.cluster_name}-sl"

  ingress_security_rules {
    protocol    = "all"
    source      = "10.0.0.0/16"
    description = "Intra-VCN NCCL traffic"
  }

  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    description = "SSH access"

    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    description = "vLLM API"

    tcp_options {
      min = 8000
      max = 8000
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
    description = "Allow all egress"
  }
}

resource "oci_core_subnet" "nccl_subnet" {
  count             = var.create_network ? 1 : 0
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.nccl_vcn[0].id
  cidr_block        = "10.0.1.0/24"
  display_name      = "${var.cluster_name}-subnet"
  dns_label         = "ncclsubnet"
  route_table_id    = oci_core_route_table.nccl_rt[0].id
  security_list_ids = [oci_core_security_list.nccl_sl[0].id]
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_ocid
}

resource "oci_core_instance" "gpu_node" {
  count               = var.node_count
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "${var.cluster_name}-node-${count.index}"
  shape               = var.instance_shape

  shape_config {
    ocpus         = var.instance_ocpus
    memory_in_gbs = var.instance_memory_gb
  }

  source_details {
    source_type             = "image"
    source_id               = var.instance_image_ocid
    boot_volume_size_in_gbs = var.boot_volume_size_gb
  }

  create_vnic_details {
    subnet_id        = var.create_network ? oci_core_subnet.nccl_subnet[0].id : var.subnet_id
    display_name     = "${var.cluster_name}-vnic-${count.index}"
    assign_public_ip = true
    hostname_label   = "nccl-node-${count.index}"
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/user_data.sh.tpl", {
      cluster_name        = var.cluster_name
      node_index          = count.index
      total_nodes         = var.node_count
      container_image_url = var.container_image_url
    }))
  }

  freeform_tags = {
    Project   = "High-Throughput-LLM-Engine"
    Cluster   = var.cluster_name
    NodeIndex = tostring(count.index)
    Role      = "worker"
  }

  lifecycle { ignore_changes = [metadata] }
}