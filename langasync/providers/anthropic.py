"""Anthropic Message Batches API adapter."""

import json
import os
from datetime import datetime
from typing import Any

import httpx
from langchain_anthropic.chat_models import _format_messages
from langchain_anthropic.output_parsers import extract_tool_calls
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langasync.core.exceptions import provider_error_handling
from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchApiJob,
    BatchResponse,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
    Provider,
)


def _to_anthropic_request(inp: LanguageModelInput, model_config: dict, custom_id: str) -> dict:
    """Convert LanguageModelInput to Anthropic batch request format."""
    # Handle PromptValue (from prompt templates)
    if isinstance(inp, PromptValue):
        inp = inp.to_messages()

    # Handle string input
    if isinstance(inp, str):
        inp = [HumanMessage(content=inp)]

    # Use LangChain's _format_messages to convert
    system, messages = _format_messages(inp)

    params = {**model_config, "messages": messages}
    if system:
        params["system"] = system

    return {"custom_id": custom_id, "params": params}


def _map_anthropic_status(processing_status: str, request_counts: dict) -> BatchStatus:
    """Map Anthropic processing_status to our BatchStatus enum."""
    if processing_status == "in_progress":
        return BatchStatus.IN_PROGRESS
    elif processing_status == "canceling":
        return BatchStatus.IN_PROGRESS
    elif processing_status == "ended":
        # Check if all succeeded or if there were failures
        succeeded = request_counts.get("succeeded", 0)
        errored = request_counts.get("errored", 0)
        canceled = request_counts.get("canceled", 0)
        expired = request_counts.get("expired", 0)

        if succeeded > 0:
            return BatchStatus.COMPLETED
        elif errored > 0:
            return BatchStatus.FAILED
        elif expired > 0:
            return BatchStatus.EXPIRED
        elif canceled > 0:
            return BatchStatus.CANCELLED
        else:
            return BatchStatus.FAILED

    return BatchStatus.PENDING


class AnthropicBatchApiAdapter(BatchApiAdapterInterface):
    """Anthropic Message Batches API adapter.

    Args:
        api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        base_url: Anthropic API base URL. Defaults to https://api.anthropic.com
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    def _get_model_config(self, language_model: LanguageModelType) -> dict[str, Any]:
        """Extract model config from LangChain model for batch request."""
        model = getattr(language_model, "model_name", None) or getattr(
            language_model, "model", "claude-sonnet-4-5-20250929"
        )
        config: dict[str, Any] = {"model": model}

        for param in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
            value = getattr(language_model, param, None)
            if value is not None:
                config[param] = value

        # max_tokens is required for Anthropic
        if "max_tokens" not in config:
            config["max_tokens"] = 1024

        config.update(getattr(language_model, "model_kwargs", {}))

        return config

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
    ) -> BatchApiJob:
        """Create a new batch job with Anthropic."""
        model_config = self._get_model_config(language_model)

        requests = [
            _to_anthropic_request(inp, model_config, str(i)) for i, inp in enumerate(inputs)
        ]

        response = await self._client.post(
            f"{self.base_url}/v1/messages/batches",
            json={"requests": requests},
        )
        response.raise_for_status()
        batch_data = response.json()

        return BatchApiJob(
            id=batch_data["id"],
            provider=Provider.ANTHROPIC,
            created_at=datetime.fromisoformat(batch_data["created_at"].replace("Z", "+00:00")),
        )

    @provider_error_handling
    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        """Get the current status of a batch job."""
        response = await self._client.get(f"{self.base_url}/v1/messages/batches/{batch_api_job.id}")
        response.raise_for_status()
        data = response.json()

        request_counts = data.get("request_counts", {})
        processing = request_counts.get("processing", 0)
        succeeded = request_counts.get("succeeded", 0)
        errored = request_counts.get("errored", 0)
        canceled = request_counts.get("canceled", 0)
        expired = request_counts.get("expired", 0)

        total = processing + succeeded + errored + canceled + expired
        completed = succeeded
        failed = errored + canceled + expired

        return BatchStatusInfo(
            status=_map_anthropic_status(data["processing_status"], request_counts),
            total=total,
            completed=completed,
            failed=failed,
        )

    @provider_error_handling
    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        """List recent batch jobs."""
        response = await self._client.get(
            f"{self.base_url}/v1/messages/batches",
            params={"limit": limit},
        )
        response.raise_for_status()
        data = response.json()

        return [
            BatchApiJob(
                id=batch["id"],
                provider=Provider.ANTHROPIC,
                created_at=datetime.fromisoformat(batch["created_at"].replace("Z", "+00:00")),
            )
            for batch in data.get("data", [])
        ]

    def _parse_result_line(self, data: dict) -> BatchResponse:
        """Parse a single line from results JSONL."""
        custom_id = data.get("custom_id", "")
        result = data.get("result", {})
        result_type = result.get("type")

        if result_type == "succeeded":
            message = result.get("message", {})
            content_blocks = message.get("content", [])

            # Match LangChain's ChatAnthropic behavior exactly
            if (
                len(content_blocks) == 1
                and content_blocks[0].get("type") == "text"
                and not content_blocks[0].get("citations")
            ):
                # Single text block, no citations -> string content
                content = content_blocks[0].get("text", "")
                ai_message = AIMessage(content=content)
            elif any(b.get("type") == "tool_use" for b in content_blocks):
                # Has tool_use -> list content with tool_calls extracted
                tool_calls = extract_tool_calls(content_blocks)
                ai_message = AIMessage(content=content_blocks, tool_calls=tool_calls)
            else:
                # Multiple blocks or citations -> list content
                ai_message = AIMessage(content=content_blocks)

            return BatchResponse(
                custom_id=custom_id,
                success=True,
                content=ai_message,
                usage=message.get("usage"),
            )
        else:
            # errored, canceled, or expired
            error = result.get("error", {"type": result_type, "message": result_type})
            return BatchResponse(
                custom_id=custom_id,
                success=False,
                error=error,
            )

    @provider_error_handling
    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        """Get results from a completed batch job."""
        # First get batch info to get results_url
        response = await self._client.get(f"{self.base_url}/v1/messages/batches/{batch_api_job.id}")
        response.raise_for_status()
        batch_data = response.json()

        results_url = batch_data.get("results_url")
        if not results_url:
            return []

        # Download results
        results_response = await self._client.get(results_url)
        results_response.raise_for_status()

        results = []
        for line in results_response.text.strip().split("\n"):
            if line:
                result = self._parse_result_line(json.loads(line))
                results.append(result)

        results.sort(key=lambda r: int(r.custom_id) if r.custom_id.isdigit() else 0)
        return results

    @provider_error_handling
    async def cancel(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        """Cancel a batch job."""
        await self._client.post(f"{self.base_url}/v1/messages/batches/{batch_api_job.id}/cancel")
        return await self.get_status(batch_api_job)
