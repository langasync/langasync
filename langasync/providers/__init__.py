"""Provider implementations for various third-party APIs."""

from langasync.core.batch_api import Provider
from langasync.providers.no_provider import NoModelBatchApiAdapter
from langasync.providers.openai_provider import OpenAIBatchApiAdapter


ADAPTER_REGISTRY: dict[Provider, type] = {
    Provider.NONE: NoModelBatchApiAdapter,
    Provider.OPENAI: OpenAIBatchApiAdapter,
    # Provider.ANTHROPIC: AnthropicBatchApiAdapter,
}
