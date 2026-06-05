terraform {
  backend "s3" {}
}

provider "aws" {
  region              = var.aws_region
  allowed_account_ids = [var.allowed_account_id]
}

module "student_stack" {
  source            = "git::https://github.com/kuleuven-realization-of-ai/shared-resources.git//terraform/modules/student-stack?ref=main"
  ec2_key_pair_name = var.ec2_key_pair_name
  ec2_instance_type = var.ec2_instance_type
}

output "ecr_repository_url" {
  value = module.student_stack.ecr_repository_url
}

output "ec2_instance_id" {
  value = module.student_stack.ec2_instance_id
}

output "ec2_instance_public_ip" {
  value = module.student_stack.ec2_instance_public_ip
}

output "ec2_instance_public_dns" {
  value = module.student_stack.ec2_instance_public_dns
}

output "s3_bucket_name" {
  value = module.student_stack.s3_bucket_name
}

output "domain_name" {
  value = module.student_stack.domain_name
}
