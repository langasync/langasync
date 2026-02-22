# AWS Bedrock Setup Guide

Bedrock batch inference requires more setup than other providers (which just need an API key). This guide walks through the full setup.

## Prerequisites

- An AWS account
- AWS CLI installed (`brew install awscli` or [install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))

## 1. Create an S3 Bucket

Bedrock uses S3 for batch input/output files. Create a bucket in your chosen region:

```bash
aws s3 mb s3://my-langasync-batch --region eu-west-2
```

Or in the AWS Console: **S3** → **Create bucket** → pick a name and region.

> **Important:** Your S3 bucket and `AWS_REGION` must be in the same region.

## 2. Create an IAM Role for Bedrock

Bedrock needs an IAM role to access your S3 bucket on your behalf.

### a. Create the role with Bedrock as trusted service

In the AWS Console: **IAM** → **Roles** → **Create role**

- **Trusted entity type:** Custom trust policy
- Paste this trust policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

- Name it something like `BedrockBatchRole`

### b. Attach S3 and model permissions

Add an inline policy to the role granting access to your bucket and model invocation:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-langasync-batch",
                "arn:aws:s3:::my-langasync-batch/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": "*"
        }
    ]
}
```

Replace `my-langasync-batch` with your bucket name. You can scope the `bedrock:InvokeModel` resource to specific model ARNs if needed.

## 3. Create an IAM User with Credentials

### a. Create the user

**IAM** → **Users** → **Create user** → name it (e.g. `langasync-batch-user`)

### b. Attach permissions

Add an inline policy with Bedrock and PassRole permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:ListModelInvocationJobs",
                "bedrock:StopModelInvocationJob"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockBatchRole"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-langasync-batch",
                "arn:aws:s3:::my-langasync-batch/*"
            ]
        }
    ]
}
```

Replace `YOUR_ACCOUNT_ID` and `my-langasync-batch` with your values.

> **Why PassRole?** When creating a batch job, your user tells Bedrock to assume the role from step 2. AWS requires explicit permission for this.

### c. Generate access keys

**IAM** → **Users** → your user → **Security credentials** → **Create access key** → **Application running outside AWS** → copy the access key ID and secret.

## 4. Enable Model Access

Anthropic models on Bedrock require a one-time use case form submission:

1. Go to **Amazon Bedrock** in the AWS Console
2. **Model catalog** (left sidebar) → search for the model you want (e.g. Claude Sonnet 4.6)
3. Click on the model → **Open in playground**
4. AWS will prompt you to fill out the **Anthropic use case details form**
5. Submit and wait ~15 minutes for approval

## 5. Configure Environment

Add these to your `.env` file or export as environment variables:

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-2
BEDROCK_S3_BUCKET=my-langasync-batch
BEDROCK_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockBatchRole
```

Or use `~/.aws/credentials` and instance profiles instead of explicit keys.

## 6. Cross-Region Inference

Bedrock batch inference requires **cross-region inference profiles**. Instead of sending requests to a single region, you use a geographic prefix (`us.`, `eu.`, `apac.`) and AWS routes to whichever region in that geography has capacity.

For example, `eu.anthropic.claude-sonnet-4-6` routes to any available EU region.

langasync handles this automatically — `settings.bedrock_region_prefix` derives the correct prefix from your `AWS_REGION`:

| Region prefix | Inference prefix | Covers |
|---------------|-----------------|--------|
| `us-*` | `us.` | US and Canada regions |
| `eu-*` | `eu.` | EU and Israel regions |
| `ap-*` | `apac.` | Asia-Pacific and Middle East regions |

> **Note:** Not all models are available in all regions for batch inference. Check [Supported models for batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html) for your region.

## 7. Minimum Batch Size

Bedrock batch inference requires a **minimum of 100 records** per job. langasync validates this before submitting and raises an error if the input list is too small. If you have fewer than 100 inputs, consider using a different provider or the standard (non-batch) API.

## 8. Run the Example

```bash
# Submit a batch job
python examples/bedrock_example.py run

# Fetch results (can be run later, even after restart)
python examples/bedrock_example.py fetch
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Bedrock batch inference requires at least 100 records` | Batch job submitted with fewer than 100 inputs | Bedrock enforces a minimum of 100 records per job (see step 7). |
| `Customer doesn't have permissions to invokeModel` | The Bedrock IAM role lacks model invocation permissions | Add `bedrock:InvokeModel` to the **role** policy (see step 2b). |
| `Could not validate ListBucket permissions for S3URI` | The Bedrock IAM role can't access the S3 bucket | Ensure the bucket name in the role policy (step 2b) matches exactly, and the S3 bucket is in the **same region** as your Bedrock endpoint (`AWS_REGION`). |
| `The provided model identifier is invalid` | Model not available in your region, or missing cross-region prefix | Check [supported models](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html) for your region. Use cross-region prefix (e.g. `eu.anthropic.claude-sonnet-4-6`). |
| `User is not authorized to perform: iam:PassRole` | IAM user missing PassRole permission | Add `iam:PassRole` policy for the Bedrock role ARN (see step 3b). |
| `Model use case details have not been submitted` | Anthropic requires use case form | Open model in Bedrock playground to trigger the form (see step 4). Wait ~15 minutes. |
| `The provided ARN is invalid` | Using short job ID instead of full ARN | Ensure you're on the latest version of langasync. |
