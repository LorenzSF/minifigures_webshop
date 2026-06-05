# Get started with Terraform and the `student-stack` module.

## What is the `student-stack` module?

To help you get started with Infrastructure-as-Code (IaC) and Terraform, we created the `student-stack`
module. This Terraform module will avoid you to have to write low-level Terraform resources definitions
which would involve concepts we didn't cover in this course.

The `student-stack` Terraform module will deploy for you the following AWS resources:

  * An EC2 instance (to deploy your application) assuming the `kulroai_ec2_role` IAM role. This role 
    allows the EC2 instance to read from ECR registries (for your Docker images) and S3 buckets (for 
    your ML models). The instance is deployed with the reference security group `sg_kulroai_student_vm` 
    attached to it. This security group allows inboud internet traffic on ports 80 (HTTP), 443 (HTTPS), and 22 (SSH).

  * An ECR registry (to store Docker images) named after the student's IAM user name. If your IAM user
    name is `student1`, the ECR registry will be named `student1` meaning you'll also need to tag your
    Docker images with `student1` as a name to be able to push them to the registry (e.g. `student1:latest`
    or `student1/app-image:latest`).

  * An S3 bucket with the named after the student's IAM user name. Just like for the ECR registry, if 
    your IAM user name is `student1`, the S3 bucket will be named `student1`. You have full access (create,
    read, update, delete objects) inside your bucket.

## How to use the `student-stack` module?

### Prerequisites.

Using the `student-stack` module is very simple considering the fact that Terraform is already installed
in your GitHub CodeSpace. Note that you will need to have a configured `aws` CLI profile on your machine 
and a SSH keypair. [Refer to the dedicated documentation page to setup your AWS credentials properly.](./aws_get_started.md)

### Deploy the `student-stack` module.

Now that you have Terraform and all the prerequisites installed, you can use them to deploy the `student-stack` 
module. To do so:
  
* Create the `infrastructure` (arbitrary name) folder in your project's root. In that folder, create 
  an empty file named `main.tf`:

  ![main_tf](./img/main_tf.png)

* The Terraform code is already present in `infrastructure/terraform/main.tf`. Keep account-specific
  values out of Git by creating local configuration files from the examples:

  ```bash
  cd infrastructure/terraform
  cp terraform.tfvars.example terraform.tfvars
  cp backend.hcl.example backend.hcl
  ```

  Fill in `terraform.tfvars` with your AWS account ID and EC2 key pair name. Fill in `backend.hcl`
  with the Terraform state bucket, state key, region, and DynamoDB lock table provided for your account.

* Now that you have the `main.tf` file, you can initialize Terraform and deploy the `student-stack` module.
  Start by initializing Terraform (make sure you have your `aws` CLI properly configured). From the
  root of your repository, run:
  ```bash
  cd infrastructure/terraform
  terraform init -backend-config=backend.hcl
  ```
  If this checkout was already initialized with another backend configuration, run:
  ```bash
  terraform init -reconfigure -backend-config=backend.hcl
  ```
  ![terraform_init](./img/tf_init.png)

* Plan the deployment of the `student-stack` module. This will allow you to observe what are the expected
  changes to be applied by Terraform. It's a verification step. Run:
  ```bash
  terraform plan
  ```

  ![terraform_plan](./img/tf_plan.png)

* Observe the output of the `terraform plan` command. If satisfying, you can deploy your infrastructure
  by running the following command. You will have to specifically type `yes` as confirmation to deploy:
  ```bash
  terraform apply
  ```

  ![terraform_apply](./img/tf_apply.png)

* The `terraform apply` command will take a bit of time to execute (the time for AWS to provision your
  resources). Once the apply is done, you can observe the outputs of the `student-stack` module in the
  terminal. You will need them later to connect to your EC2 instance and push your Docker images to
  your ECR registry. If you want to see the outputs again, you can run the following command in your
  `infrastructure` folder:
  ```bash
  terraform output
  ```

  ![terraform_output](./img/tf_apply_done.png)
  ![terraform_output](./img/tf_output.png)

## Conclusion & next steps.

You have now deployed your `student-stack` with all the AWS resources it contains. You can now use 
the outputs of the module to connect to your EC2 instance and run your Docker image from there. Refer
to the [dedicated documentation page](./3_run_your_app.md) to learn how to connect to your EC2 instance
and run your application.
