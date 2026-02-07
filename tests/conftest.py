"""Shared test fixtures."""

import pytest

from langasync.settings import LangasyncSettings


@pytest.fixture
def test_settings(tmp_path) -> LangasyncSettings:
    """Create test settings with temporary storage path."""
    return LangasyncSettings(
        openai_api_key="test-api-key",
        anthropic_api_key="test-api-key",
        openai_base_url="https://api.openai.com/v1",
        anthropic_base_url="https://api.anthropic.com",
        base_storage_path=str(tmp_path),
    )
