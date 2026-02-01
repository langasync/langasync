"""Provider implementations for various third-party APIs."""

from langasync.core.batch_api import Provider
from langasync.providers.none import NoModelBatchApiAdapter
from langasync.providers.openai import OpenAIBatchApiAdapter
from langasync.providers.anthropic import AnthropicBatchApiAdapter


ADAPTER_REGISTRY: dict[Provider, type] = {
    Provider.NONE: NoModelBatchApiAdapter,
    Provider.OPENAI: OpenAIBatchApiAdapter,
    Provider.ANTHROPIC: AnthropicBatchApiAdapter,
}
