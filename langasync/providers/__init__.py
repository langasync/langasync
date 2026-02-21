"""Provider implementations for various third-party APIs."""

from pydantic import SecretStr
from langchain_core.language_models import BaseLanguageModel

from langasync.exceptions import UnsupportedProviderError
from langasync.settings import LangasyncSettings
from langasync.providers.interface import Provider, ProviderJobAdapterInterface
from langasync.providers.none import NoModelProviderJobAdapter
from langasync.providers.openai import OpenAIProviderJobAdapter
from langasync.providers.anthropic import AnthropicProviderJobAdapter
from langasync.providers.gemini import GeminiProviderJobAdapter
from langasync.providers.bedrock import BedrockProviderJobAdapter

from typing import Callable

ADAPTER_REGISTRY: dict[Provider, Callable[[LangasyncSettings], ProviderJobAdapterInterface]] = {
    Provider.NONE: lambda settings: NoModelProviderJobAdapter(settings),
    Provider.OPENAI: lambda settings: OpenAIProviderJobAdapter(settings),
    Provider.ANTHROPIC: lambda settings: AnthropicProviderJobAdapter(settings),
    Provider.GOOGLE: lambda settings: GeminiProviderJobAdapter(settings),
    Provider.BEDROCK: lambda settings: BedrockProviderJobAdapter(settings),
}


def _extract_secret(value: SecretStr | str | None) -> str | None:
    """Extract a string from a SecretStr or plain string."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def _get_provider_and_update_settings_from_model(
    model: BaseLanguageModel | None, settings: LangasyncSettings
) -> Provider:
    """Detect the provider from a model instance and extract API key if missing from settings."""
    if model is None:
        return Provider.NONE

    lc_id = model.lc_id()
    lc_path = ".".join(lc_id).lower()

    if "openai" in lc_path:
        if settings.openai_api_key is None:
            key = _extract_secret(getattr(model, "openai_api_key", None))
            if key:
                settings.openai_api_key = key
        return Provider.OPENAI
    elif "anthropic" in lc_path:
        if settings.anthropic_api_key is None:
            key = _extract_secret(getattr(model, "anthropic_api_key", None))
            if key:
                settings.anthropic_api_key = key
        return Provider.ANTHROPIC
    elif "bedrock" in lc_path:
        # AWS credentials come from env vars / instance profiles, not from the model
        return Provider.BEDROCK
    elif "google" in lc_path or "genai" in lc_path:
        if settings.google_api_key is None:
            key = _extract_secret(getattr(model, "google_api_key", None))
            if key:
                settings.google_api_key = key
        return Provider.GOOGLE

    raise UnsupportedProviderError(f"Cannot detect provider for model: {lc_id}")


def get_adapter_from_provider(
    provider: Provider, settings: LangasyncSettings
) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a provider name."""
    adapter_constructor = ADAPTER_REGISTRY.get(provider)
    if adapter_constructor is None:
        raise UnsupportedProviderError(f"Unknown provider: {provider}")
    return adapter_constructor(settings)


def get_adapter_from_model(
    model: BaseLanguageModel | None, settings: LangasyncSettings
) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a model."""
    provider = _get_provider_and_update_settings_from_model(model, settings)
    return get_adapter_from_provider(provider, settings)
