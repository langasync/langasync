"""Anthropic API response factories.

These factories are the single source of truth for Anthropic API response structures.
When the API schema changes, update these factories and all tests will follow.

Reference: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

import json
from typing import Any


def anthropic_batch_response(
    batch_id: str = "msgbatch_abc123",
    processing_status: str = "in_progress",
    created_at: str = "2024-01-15T12:00:00Z",
    request_counts: dict[str, int] | None = None,
    results_url: str | None = None,
) -> dict[str, Any]:
    """Factory for Anthropic batch response (create/get batch).

    Used by: POST /v1/messages/batches, GET /v1/messages/batches/{id}
    """
    response = {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": processing_status,
        "created_at": created_at,
    }

    if request_counts is not None:
        response["request_counts"] = request_counts

    if results_url is not None:
        response["results_url"] = results_url

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
    return anthropic_batch_response(
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


def anthropic_result_line(
    custom_id: str,
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> dict[str, Any]:
    """Factory for a successful result line in Anthropic JSONL results.

    Used by: Results JSONL from results_url
    """
    return {
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


def anthropic_error_result_line(
    custom_id: str,
    error_type: str = "errored",
    error_message: str = "Internal error",
) -> dict[str, Any]:
    """Factory for an error result line in Anthropic JSONL results.

    Used by: Results JSONL from results_url
    """
    return {
        "custom_id": custom_id,
        "result": {
            "type": error_type,
            "error": {
                "type": "api_error",
                "message": error_message,
            },
        },
    }


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

    return {
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


def anthropic_results_jsonl(results: list[dict[str, Any]]) -> str:
    """Convert list of result dicts to JSONL string."""
    return "\n".join(json.dumps(r) for r in results)
