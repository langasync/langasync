"""Anthropic Message Batches API adapter."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)
from langchain_anthropic.chat_models import _format_messages
from langchain_anthropic.output_parsers import extract_tool_calls
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue

from langasync.exceptions import ApiTimeoutError, AuthenticationError, provider_error_handling
from langasync.settings import LangasyncSettings

from langasync.providers.interface import (
    FINISHED_STATUSES,
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
    Provider,
)


def custom_convert_to_anthropic_messages(
    inp: LanguageModelInput,
) -> tuple[str | list[dict] | None, list[dict]]:
    # Handle PromptValue (from prompt templates)
    if isinstance(inp, PromptValue):
        inp = inp.to_messages()

    # Handle string input
    if isinstance(inp, str):
        inp = [HumanMessage(content=inp)]

    # Handle single message (wrap in list)
    if isinstance(inp, BaseMessage):
        inp = [inp]

    # Use LangChain's _format_messages to convert
    system, messages = _format_messages(inp)  # type: ignore[arg-type]
    return system, messages


def _to_anthropic_request(inp: LanguageModelInput, model_config: dict, custom_id: str) -> dict:
    """Convert LanguageModelInput to Anthropic batch request format."""
    system, messages = custom_convert_to_anthropic_messages(inp)

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


class AnthropicProviderJobAdapter(ProviderJobAdapterInterface):
    """Anthropic Message Batches API adapter.

    Args:
        api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        base_url: Anthropic API base URL. Defaults to https://api.anthropic.com
    """

    def __init__(self, settings: LangasyncSettings):
        self.api_key = settings.anthropic_api_key
        self.base_url = settings.anthropic_base_url
        if not self.api_key:
            raise AuthenticationError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )
        self._client = httpx.AsyncClient(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    def _get_model_config(
        self, language_model: LanguageModelType, model_bindings: dict | None = None
    ) -> dict[str, Any]:
        """Extract model config from LangChain model for batch request."""
        model = getattr(language_model, "model_name", None) or getattr(
            language_model, "model", None
        )
        if not model:
            raise ValueError(
                "Could not determine model name from language model. "
                "Ensure your model has a 'model' or 'model_name' attribute."
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

        # Merge bindings from .bind() calls (tools, tool_choice, etc.)
        if model_bindings:
            config.update(model_bindings)

        return config

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
        model_bindings: dict | None = None,
    ) -> ProviderJob:
        """Create a new batch job with Anthropic."""
        model_config = self._get_model_config(language_model, model_bindings)

        requests = [
            _to_anthropic_request(inp, model_config, str(i)) for i, inp in enumerate(inputs)
        ]
        logger.debug(f"Submitting {len(inputs)} requests to Anthropic Message Batches API")

        response = await self._client.post(
            f"{self.base_url}/v1/messages/batches",
            json={"requests": requests},
        )
        response.raise_for_status()
        batch_data = response.json()

        logger.info(f"Anthropic batch created: {batch_data['id']}")
        return ProviderJob(
            id=batch_data["id"],
            provider=Provider.ANTHROPIC,
            created_at=datetime.fromisoformat(batch_data["created_at"].replace("Z", "+00:00")),
        )

    @provider_error_handling
    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
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
        logger.info(f"Anthropic batch {batch_api_job.id}: status={data['processing_status']}")

        return BatchStatusInfo(
            status=_map_anthropic_status(data["processing_status"], request_counts),
            total=total,
            completed=completed,
            failed=failed,
        )

    @provider_error_handling
    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        """List recent batch jobs."""
        response = await self._client.get(
            f"{self.base_url}/v1/messages/batches",
            params={"limit": limit},
        )
        response.raise_for_status()
        data = response.json()

        return [
            ProviderJob(
                id=batch["id"],
                provider=Provider.ANTHROPIC,
                created_at=datetime.fromisoformat(batch["created_at"].replace("Z", "+00:00")),
            )
            for batch in data.get("data", [])
        ]

    def _parse_result_line(self, data: dict) -> BatchItem:
        """Parse a single line from results JSONL."""
        custom_id = data.get("custom_id", "")
        result = data.get("result", {})
        result_type = result.get("type")

        if result_type == "succeeded":
            message = result.get("message", {})
            content_blocks = message.get("content", [])

            # Same 3-branch logic as ChatAnthropic._create_chat_result (response parsing)
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

            return BatchItem(
                custom_id=custom_id,
                success=True,
                content=ai_message,
                usage=message.get("usage"),
            )
        else:
            # errored, canceled, or expired
            # Error structure: result.error.error contains the actual error object
            error_wrapper = result.get("error", {})
            error = error_wrapper.get("error", {"type": result_type, "message": result_type})
            return BatchItem(
                custom_id=custom_id,
                success=False,
                error=error,
            )

    @provider_error_handling
    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        """Get results from a completed batch job."""
        logger.debug(f"Downloading results for batch {batch_api_job.id}")
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
    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Cancel a batch job and wait until cancellation completes."""
        await self._client.post(f"{self.base_url}/v1/messages/batches/{batch_api_job.id}/cancel")

        # Poll until batch reaches a terminal state
        cancel_timeout_seconds = 60
        for _ in range(cancel_timeout_seconds):
            status_info = await self.get_status(batch_api_job)
            if status_info.status in FINISHED_STATUSES:
                return BatchStatusInfo(
                    status=BatchStatus.CANCELLED,
                    total=status_info.total,
                    completed=status_info.completed,
                    failed=status_info.failed,
                )
            await asyncio.sleep(1)
        raise ApiTimeoutError(f"Cancel timed out after {cancel_timeout_seconds} seconds")
