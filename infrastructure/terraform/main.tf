terraform {
  backend "s3" {
    key            = "frescalorenzo/infrastructure.tfstate"
    bucket         = "terraform-states-616454187396"
    region         = "eu-west-3"
    dynamodb_table = "terraform-states"
  }
}

provider "aws" {
  region              = "eu-west-3"
  allowed_account_ids = ["516454187396"]
}

module "student_stack" {
  source            = "git::https://github.com/kuleuven-realization-of-ai/shared-resources.git//terraform/modules/student-stack?ref=main"
  ec2_key_pair_name = "fresca-lorenzo-key-pair"
  ec2_instance_type = "t3.small"
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