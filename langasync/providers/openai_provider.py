"""OpenAI Batch API adapter."""

import json
import os
import tempfile
from datetime import datetime
from typing import Any

import httpx
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages.utils import convert_to_openai_messages

from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchApiJob,
    BatchResponse,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
)


def _to_openai_messages(inp: LanguageModelInput) -> list[dict]:
    """Convert LanguageModelInput to OpenAI messages format."""
    result = convert_to_openai_messages(inp)
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


class OpenAIBatchApiAdapter(BatchApiAdapterInterface):
    """OpenAI Batch API adapter.

    Args:
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        base_url: OpenAI API base URL. Defaults to https://api.openai.com/v1
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    def _get_model_name(self, language_model: LanguageModelType) -> str:
        """Extract model name from LangChain model."""
        if hasattr(language_model, "model_name"):
            return language_model.model_name
        elif hasattr(language_model, "model"):
            return language_model.model
        else:
            return "gpt-4o-mini"

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

    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
    ) -> BatchApiJob:
        """Create a new batch job with OpenAI."""
        model_name = self._get_model_name(language_model)

        # Build JSONL content
        lines = []
        for i, inp in enumerate(inputs):
            messages = _to_openai_messages(inp)
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": messages,
                },
            }
            lines.append(json.dumps(request))

        jsonl_content = "\n".join(lines)

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

        return BatchApiJob(
            id=batch_data["id"],
            provider="openai",
            created_at=datetime.fromtimestamp(batch_data["created_at"]),
            metadata={"input_file_id": file_id},
        )

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        """Get the current status of a batch job."""
        response = await self._client.get(f"{self.base_url}/batches/{batch_api_job.id}")
        response.raise_for_status()
        data = response.json()

        request_counts = data.get("request_counts", {})

        return BatchStatusInfo(
            status=_map_openai_status(data["status"]),
            total=request_counts.get("total", 0),
            completed=request_counts.get("completed", 0),
            failed=request_counts.get("failed", 0),
        )

    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        """List recent batch jobs."""
        response = await self._client.get(
            f"{self.base_url}/batches",
            params={"limit": limit},
        )
        response.raise_for_status()
        data = response.json()

        return [
            BatchApiJob(
                id=batch["id"],
                provider="openai",
                created_at=datetime.fromtimestamp(batch["created_at"]),
                metadata={"input_file_id": batch.get("input_file_id")},
            )
            for batch in data.get("data", [])
        ]

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        """Get results from a completed batch job."""
        response = await self._client.get(f"{self.base_url}/batches/{batch_api_job.id}")
        response.raise_for_status()
        batch_data = response.json()

        results = []

        # Parse successful results from output file
        output_file_id = batch_data.get("output_file_id")
        if output_file_id:
            content = await self._download_file(output_file_id)
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                body = data.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                results.append(
                    BatchResponse(
                        custom_id=data.get("custom_id", ""),
                        success=True,
                        content=choices[0]["message"]["content"] if choices else None,
                        usage=body.get("usage"),
                    )
                )

        # Parse failed results from error file
        error_file_id = batch_data.get("error_file_id")
        if error_file_id:
            content = await self._download_file(error_file_id)
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                results.append(
                    BatchResponse(
                        custom_id=data.get("custom_id", ""),
                        success=False,
                        error=data.get("error"),
                    )
                )

        results.sort(key=lambda r: int(r.custom_id) if r.custom_id.isdigit() else 0)
        return results

    async def cancel(self, batch_api_job: BatchApiJob) -> bool:
        """Cancel a batch job."""
        try:
            response = await self._client.post(
                f"{self.base_url}/batches/{batch_api_job.id}/cancel"
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError:
            return False
