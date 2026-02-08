"""Tests that API keys are extracted from model objects when not set in settings."""

import pytest
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langasync.settings import LangasyncSettings
from langasync.providers import _get_provider_and_update_settings_from_model, _extract_secret
from langasync.providers.interface import Provider


@pytest.fixture
def no_key_settings(tmp_path) -> LangasyncSettings:
    """Settings with no API keys set."""
    return LangasyncSettings(
        openai_api_key=None,
        anthropic_api_key=None,
        base_storage_path=str(tmp_path),
    )


class TestOpenAIKeyExtraction:
    def test_extracts_key_from_model_when_settings_has_none(self, no_key_settings):
        model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-from-model")

        provider = _get_provider_and_update_settings_from_model(model, no_key_settings)

        assert provider == Provider.OPENAI
        assert no_key_settings.openai_api_key == "sk-from-model"

    def test_settings_key_takes_priority_over_model(self, tmp_path):
        settings = LangasyncSettings(
            openai_api_key="sk-from-settings",
            base_storage_path=str(tmp_path),
        )
        model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-from-model")

        _get_provider_and_update_settings_from_model(model, settings)

        assert settings.openai_api_key == "sk-from-settings"


class TestAnthropicKeyExtraction:
    def test_extracts_key_from_model_when_settings_has_none(self, no_key_settings):
        model = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key="sk-ant-from-model")

        provider = _get_provider_and_update_settings_from_model(model, no_key_settings)

        assert provider == Provider.ANTHROPIC
        assert no_key_settings.anthropic_api_key == "sk-ant-from-model"

    def test_settings_key_takes_priority_over_model(self, tmp_path):
        settings = LangasyncSettings(
            anthropic_api_key="sk-ant-from-settings",
            base_storage_path=str(tmp_path),
        )
        model = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key="sk-ant-from-model")

        _get_provider_and_update_settings_from_model(model, settings)

        assert settings.anthropic_api_key == "sk-ant-from-settings"


class TestNoModel:
    def test_returns_none_provider(self, no_key_settings):
        provider = _get_provider_and_update_settings_from_model(None, no_key_settings)

        assert provider == Provider.NONE
        assert no_key_settings.openai_api_key is None
        assert no_key_settings.anthropic_api_key is None
