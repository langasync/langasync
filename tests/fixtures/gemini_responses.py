"""Google Gemini Batch API response factories.

These factories are the single source of truth for Gemini Batch API response structures.

Reference: https://ai.google.dev/api/batch-api
"""

from typing import Any

from tests.fixtures.schema_validator import (
    validate_gemini_candidate,
    validate_gemini_error,
    validate_gemini_inline_response,
    validate_gemini_operation,
    validate_gemini_response,
)


def gemini_batch_response(
    batch_name: str = "batches/abc123",
    state: str = "BATCH_STATE_RUNNING",
    display_name: str = "langasync-batch",
    created_at: str = "2024-01-15T12:00:00.000000Z",
    done: bool = False,
    batch_stats: dict[str, str] | None = None,
    inlined_responses: list[dict] | None = None,
) -> dict[str, Any]:
    """Factory for Gemini batch response (create/get batch).

    Used by: POST /models/{model}:batchGenerateContent, GET /batches/{id}
    """
    response: dict[str, Any] = {
        "name": batch_name,
        "metadata": {
            "state": state,
            "displayName": display_name,
            "createTime": created_at,
            "updateTime": created_at,
        },
        "done": done,
    }

    if batch_stats is not None:
        response["metadata"]["batchStats"] = batch_stats

    validate_gemini_operation(response)

    if done and inlined_responses is not None:
        for item in inlined_responses:
            validate_gemini_inline_response(item)
        response["response"] = {
            "output": {
                "inlinedResponses": {
                    "responses": inlined_responses,
                }
            }
        }

    return response


def gemini_batch_status_response(
    batch_name: str = "batches/abc123",
    state: str = "BATCH_STATE_RUNNING",
    total: int = 100,
    succeeded: int = 50,
    failed: int = 0,
    pending: int = 50,
) -> dict[str, Any]:
    """Factory for Gemini batch status response with request counts.

    Used by: GET /batches/{id}
    """
    return gemini_batch_response(
        batch_name=batch_name,
        state=state,
        done=state in ("BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED",
                        "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"),
        batch_stats={
            "requestCount": str(total),
            "successfulRequestCount": str(succeeded),
            "failedRequestCount": str(failed),
            "pendingRequestCount": str(pending),
        },
    )


def gemini_inline_result(
    key: str,
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    """Factory for a successful inline response in Gemini batch results.

    Used by: response.output.inlinedResponses.responses[]
    """
    candidate = {
        "content": {
            "parts": [{"text": content}],
            "role": "model",
        },
        "finishReason": "STOP",
        "index": 0,
    }
    validate_gemini_candidate(candidate)

    response = {
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }
    validate_gemini_response(response)

    return {
        "metadata": {"key": key},
        "response": response,
    }


def gemini_tool_call_inline_result(
    key: str,
    tool_name: str = "get_weather",
    tool_args: dict[str, Any] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> dict[str, Any]:
    """Factory for a tool call inline response in Gemini batch results.

    Used by: response.output.inlinedResponses.responses[]
    """
    if tool_args is None:
        tool_args = {"location": "NYC"}

    candidate = {
        "content": {
            "parts": [
                {"text": "I'll check the weather."},
                {"functionCall": {"name": tool_name, "args": tool_args}},
            ],
            "role": "model",
        },
        "finishReason": "STOP",
        "index": 0,
    }
    validate_gemini_candidate(candidate)

    response = {
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }
    validate_gemini_response(response)

    return {
        "metadata": {"key": key},
        "response": response,
    }


def gemini_error_result(
    key: str,
    error_code: int = 13,
    error_message: str = "Internal error occurred.",
) -> dict[str, Any]:
    """Factory for an error inline response in Gemini batch results.

    Used by: response.output.inlinedResponses.responses[]
    """
    error = {
        "code": error_code,
        "message": error_message,
    }
    validate_gemini_error(error)

    return {
        "metadata": {"key": key},
        "error": error,
    }


def gemini_list_batches_response(
    operations: list[dict[str, Any]],
    next_page_token: str | None = None,
) -> dict[str, Any]:
    """Factory for Gemini list batches response.

    Used by: GET /batches

    Note: The REST API returns operations in LRO format ({operations: [...]}).
    No SDK type exists for this envelope — ListBatchJobsResponse uses a different
    format ({batchJobs: [...]}). Each individual operation is validated instead.
    """
    for op in operations:
        validate_gemini_operation(op)

    response: dict[str, Any] = {
        "operations": operations,
    }
    if next_page_token is not None:
        response["nextPageToken"] = next_page_token
    return response
