"""AWS Bedrock Batch Inference API response factories.

These factories produce response structures matching the Bedrock batch inference API.
Validated against botocore service model shapes via schema_validator.

Reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/
"""

import json
from typing import Any

from tests.fixtures.schema_validator import (
    validate_bedrock_create_job,
    validate_bedrock_get_job,
    validate_bedrock_list_jobs,
    validate_bedrock_model_output,
)


def bedrock_create_job_response(
    job_arn: str = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
) -> dict[str, Any]:
    """Factory for Bedrock CreateModelInvocationJob response."""
    response = {"jobArn": job_arn}
    validate_bedrock_create_job(response)
    return response


def bedrock_job_status_response(
    job_arn: str = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
    job_name: str = "langasync-test",
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    role_arn: str = "arn:aws:iam::123456789012:role/TestRole",
    status: str = "InProgress",
    submit_time: str = "2024-01-15T12:00:00Z",
    last_modified_time: str = "2024-01-15T12:05:00Z",
    end_time: str | None = None,
    message: str | None = None,
    input_s3_uri: str = "s3://test-bucket/input.jsonl",
    output_s3_uri: str = "s3://test-bucket/output/",
) -> dict[str, Any]:
    """Factory for Bedrock GetModelInvocationJob response."""
    response: dict[str, Any] = {
        "jobArn": job_arn,
        "jobName": job_name,
        "modelId": model_id,
        "roleArn": role_arn,
        "status": status,
        "submitTime": submit_time,
        "lastModifiedTime": last_modified_time,
        "inputDataConfig": {
            "s3InputDataConfig": {
                "s3Uri": input_s3_uri,
                "s3InputFormat": "JSONL",
            }
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {
                "s3Uri": output_s3_uri,
            }
        },
    }
    if end_time is not None:
        response["endTime"] = end_time
    if message is not None:
        response["message"] = message
    validate_bedrock_get_job(response)
    return response


def bedrock_list_jobs_response(
    jobs: list[dict[str, Any]],
    next_token: str | None = None,
) -> dict[str, Any]:
    """Factory for Bedrock ListModelInvocationJobs response.

    Each job dict must include jobArn, status, submitTime, and modelId.
    Required botocore fields (jobName, roleArn, inputDataConfig, outputDataConfig)
    are filled with defaults if missing.
    """
    defaults = {
        "jobName": "langasync-test",
        "roleArn": "arn:aws:iam::123456789012:role/TestRole",
        "inputDataConfig": {
            "s3InputDataConfig": {
                "s3Uri": "s3://test-bucket/input.jsonl",
                "s3InputFormat": "JSONL",
            }
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {
                "s3Uri": "s3://test-bucket/output/",
            }
        },
    }
    summaries = [{**defaults, **job} for job in jobs]
    response: dict[str, Any] = {"invocationJobSummaries": summaries}
    if next_token is not None:
        response["nextToken"] = next_token
    validate_bedrock_list_jobs(response)
    return response


def bedrock_output_line(
    record_id: str,
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    """Factory for a successful result line in Bedrock output JSONL."""
    model_output = {
        "id": f"msg_{record_id}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    validate_bedrock_model_output(model_output)
    return {
        "recordId": record_id,
        "modelInput": {},
        "modelOutput": model_output,
    }


def bedrock_error_output_line(
    record_id: str,
    error_code: str = "ThrottlingException",
    error_message: str = "Too many requests",
) -> dict[str, Any]:
    """Factory for an error result line in Bedrock output JSONL."""
    return {
        "recordId": record_id,
        "modelInput": {},
        "error": {
            "errorCode": error_code,
            "errorMessage": error_message,
        },
    }


def bedrock_model_output_error_line(
    record_id: str,
    error_code: int = 400,
    error_message: str = "URL sources are not supported",
) -> dict[str, Any]:
    """Factory for an error embedded in modelOutput (input validation failures)."""
    return {
        "recordId": record_id,
        "modelInput": {},
        "modelOutput": {
            "errorCode": error_code,
            "errorMessage": error_message,
            "expired": False,
            "retryable": False,
        },
    }


def bedrock_tool_use_output_line(
    record_id: str,
    tool_name: str = "get_weather",
    tool_id: str = "toolu_123",
    tool_input: dict[str, Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> dict[str, Any]:
    """Factory for a tool use result line in Bedrock output JSONL."""
    if tool_input is None:
        tool_input = {"location": "NYC"}

    model_output = {
        "id": f"msg_{record_id}",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather."},
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    validate_bedrock_model_output(model_output)
    return {
        "recordId": record_id,
        "modelInput": {},
        "modelOutput": model_output,
    }


def bedrock_manifest(
    total_record_count: int = 100,
    processed_record_count: int = 100,
    success_record_count: int = 95,
    error_record_count: int = 5,
    input_token_count: int = 50000,
    output_token_count: int = 25000,
) -> dict[str, Any]:
    """Factory for Bedrock manifest.json.out content."""
    return {
        "totalRecordCount": total_record_count,
        "processedRecordCount": processed_record_count,
        "successRecordCount": success_record_count,
        "errorRecordCount": error_record_count,
        "inputTokenCount": input_token_count,
        "outputTokenCount": output_token_count,
    }


def bedrock_results_jsonl(results: list[dict[str, Any]]) -> str:
    """Convert list of result dicts to JSONL string."""
    return "\n".join(json.dumps(r) for r in results)
