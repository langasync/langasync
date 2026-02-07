"""Provider implementations for various third-party APIs."""

from langchain_core.language_models import BaseLanguageModel

from langasync.exceptions import UnsupportedProviderError
from langasync.settings import LangasyncSettings
from langasync.providers.interface import Provider, ProviderJobAdapterInterface
from langasync.providers.none import NoModelProviderJobAdapter
from langasync.providers.openai import OpenAIProviderJobAdapter
from langasync.providers.anthropic import AnthropicProviderJobAdapter

from typing import Callable

ADAPTER_REGISTRY: dict[Provider, Callable[[LangasyncSettings], ProviderJobAdapterInterface]] = {
    Provider.NONE: lambda settings: NoModelProviderJobAdapter(settings),
    Provider.OPENAI: lambda settings: OpenAIProviderJobAdapter(settings),
    Provider.ANTHROPIC: lambda settings: AnthropicProviderJobAdapter(settings),
}


def get_adapter_from_provider(
    provider: Provider, settings: LangasyncSettings
) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a provider name."""
    adapter_constructor = ADAPTER_REGISTRY.get(provider)
    if adapter_constructor is None:
        raise UnsupportedProviderError(f"Unknown provider: {provider}")
    return adapter_constructor(settings)


def get_provider_from_model(model: BaseLanguageModel | None) -> Provider:
    """Detect the provider from a model instance using LangChain's lc_id."""
    if model is None:
        return Provider.NONE

    lc_id = model.lc_id()
    lc_path = ".".join(lc_id).lower()

    if "openai" in lc_path:
        return Provider.OPENAI
    elif "anthropic" in lc_path:
        return Provider.ANTHROPIC

    raise UnsupportedProviderError(f"Cannot detect provider for model: {lc_id}")


def get_adapter_from_model(
    model: BaseLanguageModel | None, settings: LangasyncSettings
) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a model."""
    provider = get_provider_from_model(model)
    return get_adapter_from_provider(provider, settings)
