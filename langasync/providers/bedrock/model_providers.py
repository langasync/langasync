"""Bedrock model provider strategies.

Each supported model family (Anthropic, Meta, DeepSeek, …) implements
BedrockModelProvider to handle its own config, input format, and output parsing.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

from langchain_anthropic.chat_models import _format_messages
from langchain_anthropic.output_parsers import extract_tool_calls
from langchain_core.messages import AIMessage, BaseMessage

from langasync.exceptions import UnsupportedProviderError
from langasync.providers.interface import BatchItem, LanguageModelType


class BedrockProviderEnum(str, Enum):
    """Supported model providers on Bedrock."""

    ANTHROPIC = "anthropic"
    # META = "meta"
    # DEEPSEEK = "deepseek"


class BedrockModelProvider(ABC):
    """Strategy for provider-specific Bedrock batch inference behavior."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The Bedrock model ID."""

    @property
    @abstractmethod
    def bedrock_provider(self) -> str:
        """The provider name string (e.g. 'anthropic')."""

    @abstractmethod
    def build_model_config(
        self,
        language_model: LanguageModelType,
        model_bindings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build provider-specific modelInput config."""

    @abstractmethod
    def create_model_input(
        self,
        messages: list[BaseMessage],
    ) -> dict[str, Any]:
        """Convert messages to provider-specific modelInput format."""

    @abstractmethod
    def parse_model_output(
        self,
        record_id: str,
        model_output: dict[str, Any],
    ) -> BatchItem:
        """Parse provider-specific model output into a BatchItem."""


class AnthropicBedrockProvider(BedrockModelProvider):

    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def bedrock_provider(self) -> str:
        return BedrockProviderEnum.ANTHROPIC

    def build_model_config(
        self,
        language_model: LanguageModelType,
        model_bindings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"anthropic_version": "bedrock-2023-05-31"}
        for param in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
            value = getattr(language_model, param, None)
            if value is not None:
                config[param] = value
        if "max_tokens" not in config:
            config["max_tokens"] = 1024
        config.update(getattr(language_model, "model_kwargs", {}))
        if model_bindings:
            config.update(model_bindings)
        return config

    def create_model_input(
        self,
        messages: list[BaseMessage],
    ) -> dict[str, Any]:
        system, formatted = _format_messages(messages)  # type: ignore[arg-type]
        model_input: dict[str, Any] = {"messages": formatted}
        if system:
            model_input["system"] = system
        return model_input

    def parse_model_output(
        self,
        record_id: str,
        model_output: dict[str, Any],
    ) -> BatchItem:
        content_blocks = model_output.get("content", [])

        if (
            len(content_blocks) == 1
            and content_blocks[0].get("type") == "text"
            and not content_blocks[0].get("citations")
        ):
            ai_message = AIMessage(content=content_blocks[0].get("text", ""))
        elif any(b.get("type") == "tool_use" for b in content_blocks):
            tool_calls = extract_tool_calls(content_blocks)
            ai_message = AIMessage(content=content_blocks, tool_calls=tool_calls)
        else:
            ai_message = AIMessage(content=content_blocks)

        return BatchItem(
            custom_id=record_id,
            success=True,
            content=ai_message,
            usage=model_output.get("usage"),
        )


BEDROCK_PROVIDERS: dict[BedrockProviderEnum, Callable[[str], BedrockModelProvider]] = {
    BedrockProviderEnum.ANTHROPIC: lambda model_id: AnthropicBedrockProvider(model_id),
}

REGION_PREFIXES = frozenset({"eu", "us", "us-gov", "apac", "sa", "amer", "global", "jp", "au"})


def _strip_region_prefix(model_id: str) -> str:
    """Strip the geo region prefix from a Bedrock model ID if present.

    Examples:
        "us.anthropic.claude-3-sonnet-20240229-v1:0" -> "anthropic.claude-3-sonnet-20240229-v1:0"
        "anthropic.claude-3-sonnet-20240229-v1:0" -> "anthropic.claude-3-sonnet-20240229-v1:0"
    """
    parts = model_id.split(".", maxsplit=2)
    if len(parts) > 1 and parts[0].lower() in REGION_PREFIXES:
        return ".".join(parts[1:])
    return model_id


def _get_provider_str(model_id: str) -> str:
    """Extract provider name from Bedrock model ID.

    Examples:
        "anthropic.claude-3-sonnet-20240229-v1:0" -> "anthropic"
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0" -> "anthropic"
        "meta.llama3-70b-instruct-v1:0" -> "meta"
    """
    return _strip_region_prefix(model_id).split(".")[0]


def get_provider(provider_str: str, model_id: str, region_prefix: str) -> BedrockModelProvider:
    """Get a BedrockModelProvider for the given provider name and model ID.

    Always sets the model ID's geo prefix to region_prefix, replacing
    any existing prefix or prepending if absent.
    """
    model_id = f"{region_prefix}.{_strip_region_prefix(model_id)}"

    try:
        key = BedrockProviderEnum(provider_str)
    except ValueError:
        raise UnsupportedProviderError(
            f"Unknown Bedrock provider: {provider_str}. "
            f"Known providers: {', '.join(p.value for p in BedrockProviderEnum)}"
        )
    factory = BEDROCK_PROVIDERS.get(key)
    if factory is None:
        raise UnsupportedProviderError(
            f"Bedrock provider not yet supported: {provider_str}. "
            f"Supported: {', '.join(p.value for p in BEDROCK_PROVIDERS)}"
        )
    return factory(model_id)


def get_provider_from_model(
    language_model: LanguageModelType,
    region_prefix: str,
) -> BedrockModelProvider:
    """Get a BedrockModelProvider from a LangChain language model."""
    model_id = (
        getattr(language_model, "model_id", None)
        or getattr(language_model, "model_name", None)
        or getattr(language_model, "model", None)
    )
    if not model_id:
        raise ValueError(
            "Could not determine model name from language model. "
            "Ensure your model has a 'model', 'model_name', or 'model_id' attribute."
        )

    provider_str = getattr(language_model, "provider", None) or _get_provider_str(model_id)

    return get_provider(provider_str, model_id, region_prefix)
