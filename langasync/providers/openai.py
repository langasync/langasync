"""OpenAI Batch API adapter."""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.messages.utils import convert_to_openai_messages

from langasync.exceptions import ApiTimeoutError, provider_error_handling, AuthenticationError
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


def _to_openai_messages(inp: LanguageModelInput) -> list[dict]:
    """Convert LanguageModelInput to OpenAI messages format."""
    result = convert_to_openai_messages(inp)  # type: ignore[arg-type]
    return result if isinstance(result, list) else [result]


def _map_openai_status(status: str) -> BatchStatus:
    """Map OpenAI batch status to our BatchStatus enum."""
    mapping = {
        "validating": BatchStatus.VALIDATING,
        "in_progress": BatchStatus.IN_PROGRESS,
        "finalizing": BatchStatus.IN_PROGRESS,
        "completed": BatchStatus.COMPLETED,
        "failed": BatchStatus.FAILED,
        "cancelled": BatchStatus.CANCELLED,
        "cancelling": BatchStatus.IN_PROGRESS,
        "expired": BatchStatus.EXPIRED,
    }
    return mapping.get(status, BatchStatus.PENDING)


class OpenAIProviderJobAdapter(ProviderJobAdapterInterface):
    """OpenAI Batch API adapter.

    Args:
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        base_url: OpenAI API base URL. Defaults to https://api.openai.com/v1
    """

    def __init__(self, settings: LangasyncSettings):
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    def _get_model_config(
        self, language_model: LanguageModelType, model_bindings: dict | None = None
    ) -> dict[str, Any]:
        """Extract model config from LangChain model for batch request body."""
        model = getattr(language_model, "model_name", None) or getattr(
            language_model, "model", None
        )
        if not model:
            raise ValueError(
                "Could not determine model name from language model. "
                "Ensure your model has a 'model' or 'model_name' attribute."
            )

        config: dict[str, Any] = {"model": model}

        # Extract OpenAI params if present
        for param in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
        ):
            value = getattr(language_model, param, None)
            if value is not None:
                config[param] = value

        # Merge any extra model_kwargs
        config.update(getattr(language_model, "model_kwargs", {}))

        # Merge bindings from .bind() calls (tools, tool_choice, etc.)
        if model_bindings:
            config.update(model_bindings)

        return config

    async def _upload_batch_file(self, jsonl_content: str) -> str:
        """Upload JSONL file to OpenAI and return file ID."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                response = await self._client.post(
                    f"{self.base_url}/files",
                    files={"file": ("batch_input.jsonl", f, "application/jsonl")},
                    data={"purpose": "batch"},
                )
            response.raise_for_status()
            return response.json()["id"]
        finally:
            os.unlink(temp_path)

    async def _download_file(self, file_id: str) -> str:
        """Download file content from OpenAI."""
        response = await self._client.get(f"{self.base_url}/files/{file_id}/content")
        response.raise_for_status()
        return response.text

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
        model_bindings: dict | None = None,
    ) -> ProviderJob:
        """Create a new batch job with OpenAI."""
        model_config = self._get_model_config(language_model, model_bindings)

        # Build JSONL content
        lines = []
        for i, inp in enumerate(inputs):
            messages = _to_openai_messages(inp)
            body = {**model_config, "messages": messages}
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(request))

        jsonl_content = "\n".join(lines)
        logger.debug(f"Uploading JSONL file ({len(inputs)} requests)")

        # Upload file
        file_id = await self._upload_batch_file(jsonl_content)

        # Create batch
        response = await self._client.post(
            f"{self.base_url}/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        response.raise_for_status()
        batch_data = response.json()

        logger.info(f"OpenAI batch created: {batch_data['id']}")
        return ProviderJob(
            id=batch_data["id"],
            provider=Provider.OPENAI,
            created_at=datetime.fromtimestamp(batch_data["created_at"]),
            metadata={"input_file_id": file_id},
        )

    @provider_error_handling
    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Get the current status of a batch job."""
        response = await self._client.get(f"{self.base_url}/batches/{batch_api_job.id}")
        response.raise_for_status()
        data = response.json()

        request_counts = data.get("request_counts", {})
        logger.info(f"OpenAI batch {batch_api_job.id}: status={data['status']}")

        return BatchStatusInfo(
            status=_map_openai_status(data["status"]),
            total=request_counts.get("total", 0),
            completed=request_counts.get("completed", 0),
            failed=request_counts.get("failed", 0),
        )

    @provider_error_handling
    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        """List recent batch jobs."""
        response = await self._client.get(
            f"{self.base_url}/batches",
            params={"limit": limit},
        )
        response.raise_for_status()
        data = response.json()

        return [
            ProviderJob(
                id=batch["id"],
                provider=Provider.OPENAI,
                created_at=datetime.fromtimestamp(batch["created_at"]),
                metadata={"input_file_id": batch.get("input_file_id")},
            )
            for batch in data.get("data", [])
        ]

    def _parse_output_line(self, data: dict) -> BatchItem:
        """Parse a single line from output/error file."""
        custom_id = data.get("custom_id", "")
        response_data = data.get("response", {})
        error_data = data.get("error")
        status_code = response_data.get("status_code", 0)

        # Check for errors (explicit error field or non-200 status)
        if error_data or status_code != 200:
            return BatchItem(
                custom_id=custom_id,
                success=False,
                error=error_data or {"message": f"Request failed with status {status_code}"},
            )

        # Parse successful response using LangChain's convert_to_messages
        # This handles tool_calls extraction automatically
        body = response_data.get("body", {})
        choices = body.get("choices", [])
        if choices:
            message_dict = choices[0].get("message", {})
            # convert_to_messages expects role/content format (OpenAI format)
            ai_message = convert_to_messages([message_dict])[0]
        else:
            ai_message = AIMessage(content="")

        return BatchItem(
            custom_id=custom_id,
            success=True,
            content=ai_message,
            usage=body.get("usage"),
        )

    @provider_error_handling
    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        """Get results from a completed batch job."""
        logger.debug(f"Downloading results for batch {batch_api_job.id}")
        response = await self._client.get(f"{self.base_url}/batches/{batch_api_job.id}")
        response.raise_for_status()
        batch_data = response.json()

        # Use dict to deduplicate by custom_id (error file takes precedence)
        results_by_id: dict[str, BatchItem] = {}

        # Parse results from output file first
        output_file_id = batch_data.get("output_file_id")
        if output_file_id:
            content = await self._download_file(output_file_id)
            for line in content.strip().split("\n"):
                if line:
                    result = self._parse_output_line(json.loads(line))
                    results_by_id[result.custom_id] = result

        # Parse results from error file (overwrites if duplicate)
        error_file_id = batch_data.get("error_file_id")
        if error_file_id:
            content = await self._download_file(error_file_id)
            for line in content.strip().split("\n"):
                if line:
                    result = self._parse_output_line(json.loads(line))
                    results_by_id[result.custom_id] = result

        results = list(results_by_id.values())
        results.sort(key=lambda r: int(r.custom_id) if r.custom_id.isdigit() else 0)
        return results

    @provider_error_handling
    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Cancel a batch job and wait until cancellation completes."""
        await self._client.post(f"{self.base_url}/batches/{batch_api_job.id}/cancel")

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
