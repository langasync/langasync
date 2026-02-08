"""Unit tests for AnthropicProviderJobAdapter with mocked httpx calls."""

import json
import re
from datetime import datetime, timezone

import pytest
from pytest_httpx import HTTPXMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langasync.providers.anthropic import AnthropicProviderJobAdapter
from langasync.providers.interface import (
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    Provider,
)
from langasync.settings import LangasyncSettings
from tests.fixtures.anthropic_responses import (
    anthropic_batch_response,
    anthropic_batch_status_response,
    anthropic_result_line,
    anthropic_error_result_line,
    anthropic_tool_use_result_line,
    anthropic_results_jsonl,
    anthropic_list_batches_response,
)

# URL patterns for matching (ignores query parameters)
BATCHES_URL = re.compile(r"https://api\.anthropic\.com/v1/messages/batches(\?.*)?$")
BATCH_ID_URL = re.compile(r"https://api\.anthropic\.com/v1/messages/batches/batch_abc123.*")


@pytest.fixture
def adapter(test_settings):
    """Create Anthropic adapter with test settings."""
    return AnthropicProviderJobAdapter(test_settings)


@pytest.fixture
def mock_model():
    """Create a real ChatAnthropic model for testing."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-3-opus-20240229", temperature=0.7, api_key="test-key")


@pytest.fixture
def sample_batch_job():
    """Create a sample ProviderJob for testing."""
    return ProviderJob(
        id="batch_abc123",
        provider=Provider.ANTHROPIC,
        created_at=datetime(2024, 1, 15, 12, 0, 0),
    )


class TestCreateBatch:
    """Test create_batch method."""

    async def test_create_batch_success(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test successful batch creation."""
        httpx_mock.add_response(
            method="POST",
            url=BATCHES_URL,
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="in_progress",
                processing=2,
            ),
        )

        inputs = ["Hello, world!", "How are you?"]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id="batch_abc123",
            provider=Provider.ANTHROPIC,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify batch request content
        request = httpx_mock.get_request()
        assert request.url.path == "/v1/messages/batches"

        body = json.loads(request.content)
        assert len(body["requests"]) == 2
        assert body["requests"][0] == {
            "custom_id": "0",
            "params": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,  # ChatAnthropic default
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
        }
        assert body["requests"][1] == {
            "custom_id": "1",
            "params": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,  # ChatAnthropic default
                "messages": [{"role": "user", "content": "How are you?"}],
            },
        }

    async def test_create_batch_with_image_input(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test batch creation with image content in messages."""
        httpx_mock.add_response(
            method="POST",
            url=BATCHES_URL,
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="in_progress",
                processing=1,
            ),
        )

        inputs = [
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "url": "https://example.com/cat.jpg"},
                    ]
                ),
            ]
        ]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id="batch_abc123",
            provider=Provider.ANTHROPIC,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify batch request content — _format_messages converts to Anthropic format
        body = json.loads(httpx_mock.get_request().content)
        assert body["requests"][0] == {
            "custom_id": "0",
            "params": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,
                "system": "You are a helpful assistant.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/cat.jpg",
                                },
                            },
                        ],
                    },
                ],
            },
        }

    async def test_create_batch_with_model_bindings(
        self, adapter, mock_model, httpx_mock: HTTPXMock
    ):
        """Test batch creation with model bindings (tools, etc.)."""
        httpx_mock.add_response(
            method="POST",
            url=BATCHES_URL,
            json=anthropic_batch_response(batch_id="batch_abc123"),
        )

        bindings = {
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {}}],
            "tool_choice": {"type": "auto"},
        }
        result = await adapter.create_batch(["Test"], mock_model, model_bindings=bindings)

        assert result == ProviderJob(
            id="batch_abc123",
            provider=Provider.ANTHROPIC,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify the batch request content
        body = json.loads(httpx_mock.get_request().content)
        assert body["requests"][0] == {
            "custom_id": "0",
            "params": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,  # ChatAnthropic default
                "tools": [
                    {"name": "get_weather", "description": "Get weather", "input_schema": {}}
                ],
                "tool_choice": {"type": "auto"},
                "messages": [{"role": "user", "content": "Test"}],
            },
        }


class TestGetStatus:
    """Test get_status method."""

    async def test_get_status_in_progress(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting status for an in-progress batch."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="in_progress",
                processing=50,
                succeeded=50,
            ),
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=100, completed=50, failed=0
        )

    async def test_get_status_completed(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting status for a completed batch."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="ended",
                succeeded=100,
            ),
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.COMPLETED, total=100, completed=100, failed=0
        )

    async def test_get_status_failed(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting status for a failed batch."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="ended",
                errored=100,
            ),
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.FAILED, total=100, completed=0, failed=100
        )

    @pytest.mark.parametrize(
        "processing_status,request_counts,expected_status",
        [
            (
                "in_progress",
                {"processing": 10, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
                BatchStatus.IN_PROGRESS,
            ),
            (
                "canceling",
                {"processing": 10, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
                BatchStatus.IN_PROGRESS,
            ),
            (
                "ended",
                {"processing": 0, "succeeded": 10, "errored": 0, "canceled": 0, "expired": 0},
                BatchStatus.COMPLETED,
            ),
            (
                "ended",
                {"processing": 0, "succeeded": 0, "errored": 10, "canceled": 0, "expired": 0},
                BatchStatus.FAILED,
            ),
            (
                "ended",
                {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 10, "expired": 0},
                BatchStatus.CANCELLED,
            ),
            (
                "ended",
                {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 10},
                BatchStatus.EXPIRED,
            ),
        ],
    )
    async def test_status_mapping(
        self,
        adapter,
        sample_batch_job,
        httpx_mock: HTTPXMock,
        processing_status,
        request_counts,
        expected_status,
    ):
        """Test all Anthropic status values map correctly."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status=processing_status,
                request_counts=request_counts,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result.status == expected_status


class TestGetResults:
    """Test get_results method."""

    async def test_get_results_success(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results from a completed batch."""
        # Mock batch status with results_url
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status="ended",
                results_url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            ),
        )

        # Mock results download
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            text=anthropic_results_jsonl(
                [
                    anthropic_result_line("0", content="Hello!"),
                    anthropic_result_line(
                        "1", content="I'm fine!", input_tokens=12, output_tokens=8
                    ),
                ]
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(content="Hello!"),
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
            BatchItem(
                custom_id="1",
                success=True,
                content=AIMessage(content="I'm fine!"),
                usage={"input_tokens": 12, "output_tokens": 8},
            ),
        ]

    async def test_get_results_with_errors(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results with some failed requests."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status="ended",
                results_url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            ),
        )

        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            text=anthropic_results_jsonl(
                [
                    anthropic_result_line("0", content="Success!"),
                    anthropic_error_result_line(
                        "1",
                        error_type="errored",
                        error_message="Rate limit exceeded",
                    ),
                ]
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(content="Success!"),
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
            BatchItem(
                custom_id="1",
                success=False,
                error={"type": "api_error", "message": "Rate limit exceeded"},
            ),
        ]

    async def test_get_results_with_tool_calls(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test getting results that contain tool calls."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status="ended",
                results_url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            ),
        )

        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123/results",
            text=anthropic_results_jsonl(
                [
                    anthropic_tool_use_result_line(
                        "0",
                        tool_name="get_weather",
                        tool_id="call_123",
                        tool_input={"location": "NYC"},
                    ),
                ]
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(
                    content=[
                        {"type": "text", "text": "I'll check the weather."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"location": "NYC"},
                        },
                    ],
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                            "id": "call_123",
                            "type": "tool_call",
                        }
                    ],
                ),
                usage={"input_tokens": 10, "output_tokens": 20},
            ),
        ]

    async def test_get_results_no_results_url(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test getting results when no results_url is present."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status="in_progress",
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == []


class TestListBatches:
    """Test list_batches method."""

    async def test_list_batches(self, adapter, httpx_mock: HTTPXMock):
        """Test listing batches."""
        httpx_mock.add_response(
            method="GET",
            url=BATCHES_URL,
            json=anthropic_list_batches_response(
                [
                    anthropic_batch_response(batch_id="batch_abc123"),
                    anthropic_batch_response(
                        batch_id="batch_def456",
                        created_at="2024-01-15T12:01:40Z",
                    ),
                ]
            ),
        )

        results = await adapter.list_batches(limit=10)

        assert results == [
            ProviderJob(
                id="batch_abc123",
                provider=Provider.ANTHROPIC,
                created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            ProviderJob(
                id="batch_def456",
                provider=Provider.ANTHROPIC,
                created_at=datetime(2024, 1, 15, 12, 1, 40, tzinfo=timezone.utc),
            ),
        ]

        # Verify limit was passed
        request = httpx_mock.get_request()
        assert "limit=10" in str(request.url)


class TestCancel:
    """Test cancel method."""

    async def test_cancel_batch(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test cancelling a batch waits until terminal state."""
        # Mock cancel request
        httpx_mock.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123/cancel",
            json=anthropic_batch_response(
                batch_id="batch_abc123",
                processing_status="canceling",
            ),
        )

        # Mock get_status calls - first canceling, then ended
        # Some requests may have succeeded before cancel took effect
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="canceling",
                processing=50,
                succeeded=50,
            ),
        )
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="ended",
                succeeded=50,
                canceled=50,
            ),
        )

        result = await adapter.cancel(sample_batch_job)

        # Status is always CANCELLED when cancel() is called, regardless of succeeded requests
        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=100,
            completed=50,
            failed=50,
        )

        # Verify cancel was called then status polled twice
        requests = httpx_mock.get_requests()
        assert requests[0].url.path == "/v1/messages/batches/batch_abc123/cancel"
        assert requests[1].url.path == "/v1/messages/batches/batch_abc123"
        assert requests[2].url.path == "/v1/messages/batches/batch_abc123"

    async def test_cancel_completes_to_cancelled(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test that status transitions from canceling to cancelled."""
        # First call: canceling
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="canceling",
                processing=100,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=100, completed=0, failed=0
        )

        # Second call: ended with all canceled
        # Note: status mapping prioritizes succeeded > errored > expired > canceled
        httpx_mock.add_response(
            method="GET",
            url="https://api.anthropic.com/v1/messages/batches/batch_abc123",
            json=anthropic_batch_status_response(
                batch_id="batch_abc123",
                processing_status="ended",
                canceled=100,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED, total=100, completed=0, failed=100
        )


class TestAdapterInit:
    """Test adapter initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        settings = LangasyncSettings(anthropic_api_key="sk-test123")
        adapter = AnthropicProviderJobAdapter(settings)
        assert adapter.api_key == "sk-test123"
        assert adapter.base_url == "https://api.anthropic.com"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        settings = LangasyncSettings(
            anthropic_api_key="sk-test123",
            anthropic_base_url="https://custom.api.com/",
        )
        adapter = AnthropicProviderJobAdapter(settings)
        assert adapter.base_url == "https://custom.api.com/"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        settings = LangasyncSettings(anthropic_api_key=None)
        with pytest.raises(Exception, match="Anthropic API key required"):
            AnthropicProviderJobAdapter(settings)

    def test_init_from_settings(self):
        """Test initialization from settings object."""
        settings = LangasyncSettings(anthropic_api_key="sk-from-settings")
        adapter = AnthropicProviderJobAdapter(settings)
        assert adapter.api_key == "sk-from-settings"
