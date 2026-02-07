"""OpenAI API response factories.

These factories are the single source of truth for OpenAI API response structures.
When the SDK updates, validation will fail if our factories produce incompatible structures.

Reference: https://platform.openai.com/docs/api-reference/batch
"""

import json
from typing import Any

from tests.fixtures.schema_validator import (
    validate_openai_batch,
    validate_openai_file,
    validate_openai_chat_completion,
    validate_openai_batch_error,
    validate_openai_error_object,
)


def openai_file_upload_response(
    file_id: str = "file-abc123",
    filename: str = "batch_input.jsonl",
    purpose: str = "batch",
    bytes_size: int = 1024,
    created_at: int = 1705320000,
    status: str = "processed",
) -> dict[str, Any]:
    """Factory for OpenAI file upload response.

    Used by: POST /v1/files
    """
    response = {
        "id": file_id,
        "object": "file",
        "bytes": bytes_size,
        "created_at": created_at,
        "filename": filename,
        "purpose": purpose,
        "status": status,
    }
    validate_openai_file(response)
    return response


def openai_batch_response(
    batch_id: str = "batch_abc123",
    status: str = "in_progress",
    input_file_id: str = "file-abc123",
    output_file_id: str | None = None,
    error_file_id: str | None = None,
    created_at: int = 1705320000,
    request_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Factory for OpenAI batch response (create/get batch).

    Used by: POST /v1/batches, GET /v1/batches/{id}
    """
    response = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": input_file_id,
        "completion_window": "24h",
        "status": status,
        "created_at": created_at,
    }

    if output_file_id is not None:
        response["output_file_id"] = output_file_id

    if error_file_id is not None:
        response["error_file_id"] = error_file_id

    if request_counts is not None:
        response["request_counts"] = request_counts

    validate_openai_batch(response)
    return response


def openai_batch_status_response(
    batch_id: str = "batch_abc123",
    status: str = "in_progress",
    total: int = 100,
    completed: int = 50,
    failed: int = 0,
    input_file_id: str = "file-abc123",
    output_file_id: str | None = None,
    error_file_id: str | None = None,
) -> dict[str, Any]:
    """Factory for OpenAI batch status response with request counts.

    Used by: GET /v1/batches/{id}
    """
    response = openai_batch_response(
        batch_id=batch_id,
        status=status,
        input_file_id=input_file_id,
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        request_counts={
            "total": total,
            "completed": completed,
            "failed": failed,
        },
    )
    validate_openai_batch(response)
    return response


def openai_output_line(
    custom_id: str,
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    status_code: int = 200,
    created: int = 1705320000,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """Factory for a successful output line in OpenAI JSONL results.

    Used by: Output JSONL from output_file_id
    """
    body = {
        "id": f"chatcmpl-{custom_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    validate_openai_chat_completion(body)
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "body": body,
        },
    }


def openai_error_output_line(
    custom_id: str,
    error_message: str = "Rate limit exceeded",
    error_type: str = "rate_limit_error",
    status_code: int = 429,
) -> dict[str, Any]:
    """Factory for an error in the output file (non-200 response).

    Used by: Output JSONL from output_file_id
    """
    error = {
        "message": error_message,
        "type": error_type,
    }
    validate_openai_error_object(error)
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "body": {"error": error},
        },
    }


def openai_error_file_line(
    custom_id: str,
    error_message: str = "Rate limit exceeded",
    error_code: str = "rate_limit_exceeded",
) -> dict[str, Any]:
    """Factory for an error line in the error file.

    Used by: Error JSONL from error_file_id
    """
    error = {
        "message": error_message,
        "code": error_code,
    }
    validate_openai_batch_error(error)
    return {
        "custom_id": custom_id,
        "error": error,
    }


def openai_tool_call_output_line(
    custom_id: str,
    tool_name: str = "get_weather",
    tool_id: str = "call_123",
    arguments: str = '{"location": "NYC"}',
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    created: int = 1705320000,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """Factory for a tool call output line in OpenAI JSONL results.

    Used by: Output JSONL from output_file_id
    """
    body = {
        "id": f"chatcmpl-{custom_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    validate_openai_chat_completion(body)
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": body,
        },
    }


def openai_results_jsonl(results: list[dict[str, Any]]) -> str:
    """Convert list of result dicts to JSONL string."""
    return "\n".join(json.dumps(r) for r in results)


def openai_list_batches_response(
    batches: list[dict[str, Any]],
    has_more: bool = False,
    first_id: str | None = None,
    last_id: str | None = None,
) -> dict[str, Any]:
    """Factory for OpenAI list batches response.

    Used by: GET /v1/batches
    """
    response: dict[str, Any] = {
        "object": "list",
        "data": batches,
        "has_more": has_more,
    }
    if first_id is not None:
        response["first_id"] = first_id
    if last_id is not None:
        response["last_id"] = last_id
    return response
