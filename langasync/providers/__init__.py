"""Provider implementations for various third-party APIs."""

from enum import Enum

from langasync.providers.no_provider import NoModelBatchApiAdapter
from langasync.providers.openai_provider import OpenAIBatchApiAdapter


class Provider(str, Enum):
    """Supported batch API providers."""

    NONE = "none"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


ADAPTER_REGISTRY: dict[Provider, type] = {
    Provider.NONE: NoModelBatchApiAdapter,
    Provider.OPENAI: OpenAIBatchApiAdapter,
    # Provider.ANTHROPIC: AnthropicBatchApiAdapter,
}
