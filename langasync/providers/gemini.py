"""Google Gemini Batch API adapter."""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)
from langchain_core.language_models import LanguageModelInput
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.prompt_values import PromptValue

from langasync.exceptions import ApiTimeoutError, AuthenticationError, provider_error_handling
from langasync.settings import LangasyncSettings
from langasync.utils import generate_uuid
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


def _guess_mime_type(url: str) -> str:
    """Guess MIME type from URL extension."""
    path = url.lower().split("?")[0].split("#")[0]
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".gif"):
        return "image/gif"
    if path.endswith(".webp"):
        return "image/webp"
    if path.endswith(".pdf"):
        return "application/pdf"
    return "image/jpeg"


def _convert_content_part(part: dict) -> dict:
    """Convert a single LangChain content part dict to Gemini format."""
    part_type = part.get("type")

    if part_type == "text":
        return {"text": part["text"]}

    if part_type == "image":
        url = part["url"]
        if url.startswith("data:"):
            # data:image/jpeg;base64,/9j/4AAQ...
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inline_data": {"mime_type": mime_type, "data": data}}
        return {"file_data": {"mime_type": _guess_mime_type(url), "file_uri": url}}

    if part_type == "image_url":
        url = part.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inline_data": {"mime_type": mime_type, "data": data}}
        return {"file_data": {"mime_type": _guess_mime_type(url), "file_uri": url}}

    if part_type == "file":
        return {"inline_data": {"mime_type": part["mime_type"], "data": part["base64"]}}

    # Already in Gemini format or unknown — pass through
    return part


def _convert_tools(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tools to Gemini function_declarations format."""
    declarations = []
    for tool in openai_tools:
        if tool.get("type") == "function":
            func = tool["function"]
            decl: dict[str, Any] = {"name": func["name"]}
            if func.get("description"):
                decl["description"] = func["description"]
            if func.get("parameters"):
                decl["parameters"] = func["parameters"]
            declarations.append(decl)
    if not declarations:
        return []
    return [{"function_declarations": declarations}]


def _message_content_to_parts(content: str | list) -> list[dict]:
    """Convert LangChain message content to Gemini parts dicts."""
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append({"text": p})
        elif isinstance(p, dict):
            parts.append(_convert_content_part(p))
        else:
            parts.append(p)
    return parts


def _convert_to_gemini_messages(
    inp: LanguageModelInput,
) -> tuple[dict | None, list[dict]]:
    """Convert LangChain input to Gemini API format (system instruction + contents).

    Handles: SystemMessage, HumanMessage, AIMessage (with tool_calls), ToolMessage.
    Produces plain dicts matching the Gemini REST API schema directly.
    """
    # Normalize input to list of messages
    messages: list[BaseMessage]
    if isinstance(inp, PromptValue):
        messages = inp.to_messages()
    elif isinstance(inp, str):
        messages = [HumanMessage(content=inp)]
    elif isinstance(inp, BaseMessage):
        messages = [inp]
    else:
        messages = list(inp)  # type: ignore[arg-type]

    system_instruction: dict | None = None
    contents: list[dict] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            parts = _message_content_to_parts(message.content)
            if system_instruction is None:
                system_instruction = {"parts": parts}
            else:
                system_instruction["parts"].extend(parts)

        elif isinstance(message, HumanMessage):
            contents.append(
                {
                    "role": "user",
                    "parts": _message_content_to_parts(message.content),
                }
            )

        elif isinstance(message, AIMessage):
            if message.tool_calls:
                ai_parts: list[dict] = []
                # Include any text content before tool calls
                if message.content:
                    ai_parts.extend(_message_content_to_parts(message.content))
                for tc in message.tool_calls:
                    ai_parts.append({"functionCall": {"name": tc["name"], "args": tc["args"]}})
                contents.append({"role": "model", "parts": ai_parts})
            else:
                contents.append(
                    {
                        "role": "model",
                        "parts": _message_content_to_parts(message.content),
                    }
                )

        elif isinstance(message, ToolMessage):
            # Gemini expects tool responses as user role with functionResponse
            tool_response: Any = message.content
            if isinstance(tool_response, str):
                try:
                    tool_response = json.loads(tool_response)
                except json.JSONDecodeError:
                    tool_response = {"output": tool_response}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": message.name or "",
                                "response": tool_response,
                            }
                        }
                    ],
                }
            )

        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return system_instruction, contents


def _map_gemini_status(state: str) -> BatchStatus:
    """Map Gemini batch state to our BatchStatus enum."""
    mapping = {
        "BATCH_STATE_PENDING": BatchStatus.PENDING,
        "BATCH_STATE_RUNNING": BatchStatus.IN_PROGRESS,
        "BATCH_STATE_SUCCEEDED": BatchStatus.COMPLETED,
        "BATCH_STATE_FAILED": BatchStatus.FAILED,
        "BATCH_STATE_CANCELLED": BatchStatus.CANCELLED,
        "BATCH_STATE_EXPIRED": BatchStatus.EXPIRED,
    }
    return mapping.get(state, BatchStatus.PENDING)


class GeminiProviderJobAdapter(ProviderJobAdapterInterface):
    """Google Gemini Batch API adapter.

    Uses the direct Gemini API (generativelanguage.googleapis.com),
    not Vertex AI. Requires a Google API key.
    """

    def __init__(self, settings: LangasyncSettings):
        self.api_key = settings.google_api_key
        self.base_url = settings.google_base_url
        if not self.api_key:
            raise AuthenticationError(
                "Google API key required. Set GOOGLE_API_KEY or pass api_key."
            )
        self._client = httpx.AsyncClient(
            headers={
                "x-goog-api-key": self.api_key,
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    def _get_model_config(
        self, language_model: LanguageModelType, model_bindings: dict | None = None
    ) -> dict[str, Any]:
        """Extract model config from LangChain model for batch request."""
        config: dict[str, Any] = {}

        for param in ("temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences"):
            value = getattr(language_model, param, None)
            if value is not None:
                config[param] = value

        config.update(getattr(language_model, "model_kwargs", {}))

        if model_bindings:
            config.update(model_bindings)

        return config

    def _get_model_name(self, language_model: LanguageModelType) -> str:
        """Extract model name from LangChain model, ensuring models/ prefix."""
        model = getattr(language_model, "model_name", None) or getattr(
            language_model, "model", None
        )
        if not model:
            raise ValueError(
                "Could not determine model name from language model. "
                "Ensure your model has a 'model' or 'model_name' attribute."
            )
        if not model.startswith("models/"):
            model = f"models/{model}"
        return model

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
        model_bindings: dict | None = None,
    ) -> ProviderJob:
        """Create a new batch job with Gemini."""
        model_name = self._get_model_name(language_model)
        model_config = self._get_model_config(language_model, model_bindings)

        # Extract tools from config — they go at request level, not in generation_config
        tools_config = None
        raw_tools = model_config.pop("tools", None)
        if raw_tools:
            tools_config = _convert_tools(raw_tools)
        tool_choice = model_config.pop("tool_choice", None)

        requests = []
        for i, inp in enumerate(inputs):
            system_instruction, contents = _convert_to_gemini_messages(inp)
            request: dict[str, Any] = {"contents": contents}
            if system_instruction:
                request["system_instruction"] = system_instruction
            if model_config:
                request["generation_config"] = model_config
            if tools_config:
                request["tools"] = tools_config
            if tool_choice:
                request["tool_config"] = {"function_calling_config": {"mode": tool_choice}}
            # key=str(i) correlates responses back to inputs by submission order
            requests.append({"request": request, "metadata": {"key": str(i)}})

        logger.debug(f"Submitting {len(inputs)} requests to Gemini Batch API")

        response = await self._client.post(
            f"{self.base_url}/{model_name}:batchGenerateContent",
            json={
                "batch": {
                    "display_name": "langasync-batch",
                    "input_config": {
                        "requests": {
                            "requests": requests,
                        }
                    },
                }
            },
        )
        response.raise_for_status()
        batch_data = response.json()

        batch_name = batch_data["name"]
        created_at = batch_data.get("metadata", {}).get("createTime")

        logger.info(f"Gemini batch created: {batch_name}")
        return ProviderJob(
            id=batch_name,
            provider=Provider.GOOGLE,
            created_at=(
                datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_at
                else datetime.now()
            ),
        )

    @provider_error_handling
    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Get the current status of a batch job."""
        response = await self._client.get(f"{self.base_url}/{batch_api_job.id}")
        response.raise_for_status()
        data = response.json()

        metadata = data.get("metadata", {})
        state = metadata.get("state", "BATCH_STATE_UNSPECIFIED")
        batch_stats = metadata.get("batchStats", {})

        total = int(batch_stats.get("requestCount", 0))
        completed = int(batch_stats.get("successfulRequestCount", 0))
        failed = int(batch_stats.get("failedRequestCount", 0))

        logger.info(f"Gemini batch {batch_api_job.id}: state={state}")

        return BatchStatusInfo(
            status=_map_gemini_status(state),
            total=total,
            completed=completed,
            failed=failed,
        )

    @provider_error_handling
    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        """List recent batch jobs."""
        response = await self._client.get(
            f"{self.base_url}/batches",
            params={"pageSize": limit},
        )
        response.raise_for_status()
        data = response.json()

        return [
            ProviderJob(
                id=op["name"],
                provider=Provider.GOOGLE,
                created_at=(
                    datetime.fromisoformat(
                        op.get("metadata", {}).get("createTime", "").replace("Z", "+00:00")
                    )
                    if op.get("metadata", {}).get("createTime")
                    else datetime.now()
                ),
            )
            for op in data.get("operations", [])
        ]

    def _parse_inline_response(self, item: dict) -> BatchItem:
        """Parse a single inline response from batch results."""
        key = item.get("metadata", {}).get("key", "")

        error = item.get("error")
        if error:
            return BatchItem(
                custom_id=key,
                success=False,
                error=error,
            )

        response_data = item.get("response", {})
        candidates = response_data.get("candidates", [])

        if not candidates:
            return BatchItem(
                custom_id=key,
                success=False,
                error={"message": "No candidates in response"},
            )

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        # Match LangChain's ChatGoogleGenerativeAI behavior:
        # single text -> string, tool calls -> extracted, multiple parts -> list
        tool_calls: list[ToolCall] = []
        for part in content_parts:
            fc = part.get("functionCall")
            if fc:
                tool_calls.append(
                    ToolCall(
                        name=fc["name"],
                        args=fc.get("args", {}),
                        id=generate_uuid(),
                    )
                )

        if tool_calls:
            ai_message = AIMessage(content=content_parts, tool_calls=tool_calls)
        elif len(content_parts) == 1 and "text" in content_parts[0]:
            ai_message = AIMessage(content=content_parts[0]["text"])
        else:
            ai_message = AIMessage(content=content_parts)

        usage_metadata = response_data.get("usageMetadata", {})
        usage = None
        if usage_metadata:
            usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            }

        return BatchItem(
            custom_id=key,
            success=True,
            content=ai_message,
            usage=usage,
        )

    @provider_error_handling
    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        """Get results from a completed batch job."""
        logger.debug(f"Downloading results for batch {batch_api_job.id}")
        response = await self._client.get(f"{self.base_url}/{batch_api_job.id}")
        response.raise_for_status()
        batch_data = response.json()

        inlined = batch_data.get("response", {}).get("inlinedResponses", {})
        responses = inlined.get("inlinedResponses", [])

        if not responses:
            return []

        results = [self._parse_inline_response(item) for item in responses]
        # Restore submission order — responses may arrive out of order
        results.sort(key=lambda r: int(r.custom_id) if r.custom_id.isdigit() else 0)
        return results

    @provider_error_handling
    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Cancel a batch job and wait until cancellation completes."""
        await self._client.post(f"{self.base_url}/{batch_api_job.id}:cancel")

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
