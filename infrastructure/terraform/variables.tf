variable "aws_region" {
  description = "AWS region used for the student stack."
  type        = string
  default     = "eu-west-3"
}

variable "allowed_account_id" {
  description = "AWS account ID Terraform is allowed to manage."
  type        = string

  validation {
    condition     = can(regex("^\\d{12}$", var.allowed_account_id))
    error_message = "allowed_account_id must be a 12-digit AWS account ID."
  }
}

variable "ec2_key_pair_name" {
  description = "Name of the existing EC2 key pair used to SSH into the deployed instance."
  type        = string
}

variable "ec2_instance_type" {
  description = "EC2 instance type for the deployed application host."
  type        = string
  default     = "t3.small"
}
