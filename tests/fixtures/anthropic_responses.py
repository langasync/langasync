"""Anthropic API response factories.

These factories are the single source of truth for Anthropic API response structures.
When the SDK updates, validation will fail if our factories produce incompatible structures.

Reference: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

import json
from typing import Any

from tests.fixtures.schema_validator import validate_anthropic_batch, validate_anthropic_result_line


def anthropic_batch_response(
    batch_id: str = "msgbatch_abc123",
    processing_status: str = "in_progress",
    created_at: str = "2024-01-15T12:00:00Z",
    expires_at: str = "2024-01-16T12:00:00Z",
    request_counts: dict[str, int] | None = None,
    results_url: str | None = None,
) -> dict[str, Any]:
    """Factory for Anthropic batch response (create/get batch).

    Used by: POST /v1/messages/batches, GET /v1/messages/batches/{id}
    """
    response: dict[str, Any] = {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": processing_status,
        "created_at": created_at,
        "expires_at": expires_at,
        "request_counts": request_counts
        or {
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
    }

    if results_url is not None:
        response["results_url"] = results_url

    validate_anthropic_batch(response)
    return response


def anthropic_batch_status_response(
    batch_id: str = "msgbatch_abc123",
    processing_status: str = "in_progress",
    processing: int = 0,
    succeeded: int = 0,
    errored: int = 0,
    canceled: int = 0,
    expired: int = 0,
    results_url: str | None = None,
) -> dict[str, Any]:
    """Factory for Anthropic batch status response with request counts.

    Used by: GET /v1/messages/batches/{id}
    """
    response = anthropic_batch_response(
        batch_id=batch_id,
        processing_status=processing_status,
        request_counts={
            "processing": processing,
            "succeeded": succeeded,
            "errored": errored,
            "canceled": canceled,
            "expired": expired,
        },
        results_url=results_url,
    )
    validate_anthropic_batch(response)
    return response


def anthropic_result_line(
    custom_id: str,
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    """Factory for a successful result line in Anthropic JSONL results.

    Used by: Results JSONL from results_url
    """
    response = {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": f"msg_{custom_id}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "model": "claude-3-opus-20240229",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        },
    }
    validate_anthropic_result_line(response)
    return response


def anthropic_error_result_line(
    custom_id: str,
    error_type: str = "errored",
    error_message: str = "Internal error",
) -> dict[str, Any]:
    """Factory for an error result line in Anthropic JSONL results.

    Used by: Results JSONL from results_url
    """
    response = {
        "custom_id": custom_id,
        "result": {
            "type": error_type,
            "error": {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error_message,
                },
            },
        },
    }
    validate_anthropic_result_line(response)
    return response


def anthropic_tool_use_result_line(
    custom_id: str,
    tool_name: str = "get_weather",
    tool_id: str = "toolu_123",
    tool_input: dict[str, Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> dict[str, Any]:
    """Factory for a tool use result line in Anthropic JSONL results.

    Used by: Results JSONL from results_url
    """
    if tool_input is None:
        tool_input = {"location": "NYC"}

    response = {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": f"msg_{custom_id}",
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
                "model": "claude-3-opus-20240229",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        },
    }
    validate_anthropic_result_line(response)
    return response


def anthropic_results_jsonl(results: list[dict[str, Any]]) -> str:
    """Convert list of result dicts to JSONL string."""
    return "\n".join(json.dumps(r) for r in results)


def anthropic_list_batches_response(
    batches: list[dict[str, Any]],
    has_more: bool = False,
    first_id: str | None = None,
    last_id: str | None = None,
) -> dict[str, Any]:
    """Factory for Anthropic list batches response.

    Used by: GET /v1/messages/batches
    """
    response: dict[str, Any] = {
        "data": batches,
        "has_more": has_more,
    }
    if first_id is not None:
        response["first_id"] = first_id
    if last_id is not None:
        response["last_id"] = last_id
    return response
