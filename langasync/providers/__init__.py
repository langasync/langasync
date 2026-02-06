"""Provider implementations for various third-party APIs."""

from langasync.providers.interface import Provider
from langasync.providers.none import NoModelProviderJobAdapter
from langasync.providers.openai import OpenAIProviderJobAdapter
from langasync.providers.anthropic import AnthropicProviderJobAdapter


ADAPTER_REGISTRY: dict[Provider, type] = {
    Provider.NONE: NoModelProviderJobAdapter,
    Provider.OPENAI: OpenAIProviderJobAdapter,
    Provider.ANTHROPIC: AnthropicProviderJobAdapter,
}
