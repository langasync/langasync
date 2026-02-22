"""Unit tests for GeminiProviderJobAdapter with mocked httpx calls."""

import json
import re
from datetime import datetime, timezone

import pytest
from pytest_httpx import HTTPXMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langasync.providers.gemini import GeminiProviderJobAdapter
from langasync.providers.interface import (
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    Provider,
)
from langasync.settings import LangasyncSettings
from tests.fixtures.gemini_responses import (
    gemini_batch_response,
    gemini_batch_status_response,
    gemini_inline_result,
    gemini_error_result,
    gemini_tool_call_inline_result,
    gemini_list_batches_response,
)

# URL patterns for matching
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
BATCHES_URL = re.compile(rf"{re.escape(BASE_URL)}/batches(\?.*)?$")
BATCH_ID_URL = re.compile(rf"{re.escape(BASE_URL)}/batches/abc123.*")
BATCH_GENERATE_URL = re.compile(rf"{re.escape(BASE_URL)}/models/.*:batchGenerateContent")


@pytest.fixture
def adapter(test_settings):
    """Create Gemini adapter with test settings."""
    test_settings.google_api_key = "test-google-api-key"
    return GeminiProviderJobAdapter(test_settings)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.model = "gemini-2.5-flash"
    model.model_name = None
    model.temperature = 0.7
    model.max_output_tokens = None
    model.top_p = None
    model.top_k = None
    model.stop_sequences = None
    model.model_kwargs = {}
    return model


@pytest.fixture
def sample_batch_job():
    """Create a sample ProviderJob for testing."""
    return ProviderJob(
        id="batches/abc123",
        provider=Provider.GOOGLE,
        created_at=datetime(2024, 1, 15, 12, 0, 0),
    )


class TestCreateBatch:
    """Test create_batch method."""

    async def test_create_batch_success(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test successful batch creation."""
        httpx_mock.add_response(
            method="POST",
            url=BATCH_GENERATE_URL,
            json=gemini_batch_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_PENDING",
            ),
        )

        inputs = ["Hello, world!", "How are you?"]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id="batches/abc123",
            provider=Provider.GOOGLE,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify batch request content
        request = httpx_mock.get_request()
        assert ":batchGenerateContent" in str(request.url)

        body = json.loads(request.content)
        batch = body["batch"]
        requests = batch["input_config"]["requests"]["requests"]
        assert len(requests) == 2
        assert requests[0]["metadata"]["key"] == "0"
        assert requests[0]["request"]["contents"][0]["parts"][0]["text"] == "Hello, world!"
        assert requests[1]["metadata"]["key"] == "1"
        assert requests[1]["request"]["contents"][0]["parts"][0]["text"] == "How are you?"

    async def test_create_batch_with_image_input(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test batch creation with image content in messages."""
        httpx_mock.add_response(
            method="POST",
            url=BATCH_GENERATE_URL,
            json=gemini_batch_response(batch_name="batches/abc123"),
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
            id="batches/abc123",
            provider=Provider.GOOGLE,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify image content is converted to Gemini format
        body = json.loads(httpx_mock.get_request().content)
        request_data = body["batch"]["input_config"]["requests"]["requests"][0]["request"]
        assert request_data["system_instruction"] == {
            "parts": [{"text": "You are a helpful assistant."}]
        }
        assert request_data["contents"] == [
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this image."},
                    {
                        "file_data": {
                            "mime_type": "image/jpeg",
                            "file_uri": "https://example.com/cat.jpg",
                        }
                    },
                ],
            }
        ]

    async def test_create_batch_with_system_message(
        self, adapter, mock_model, httpx_mock: HTTPXMock
    ):
        """Test batch creation with system message."""
        httpx_mock.add_response(
            method="POST",
            url=BATCH_GENERATE_URL,
            json=gemini_batch_response(batch_name="batches/abc123"),
        )

        inputs = [
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello!"),
            ]
        ]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id="batches/abc123",
            provider=Provider.GOOGLE,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify system message is sent as system_instruction
        body = json.loads(httpx_mock.get_request().content)
        request_data = body["batch"]["input_config"]["requests"]["requests"][0]["request"]
        assert request_data["system_instruction"] == {
            "parts": [{"text": "You are a helpful assistant."}]
        }
        # Only the human message is in contents
        assert request_data["contents"] == [{"role": "user", "parts": [{"text": "Hello!"}]}]

    async def test_create_batch_with_model_bindings(
        self, adapter, mock_model, httpx_mock: HTTPXMock
    ):
        """Test batch creation with model bindings."""
        httpx_mock.add_response(
            method="POST",
            url=BATCH_GENERATE_URL,
            json=gemini_batch_response(batch_name="batches/abc123"),
        )

        bindings = {"temperature": 0.9, "top_k": 40}
        result = await adapter.create_batch(["Test"], mock_model, model_bindings=bindings)

        assert result == ProviderJob(
            id="batches/abc123",
            provider=Provider.GOOGLE,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Verify model bindings are applied to generation_config
        body = json.loads(httpx_mock.get_request().content)
        config = body["batch"]["input_config"]["requests"]["requests"][0]["request"][
            "generation_config"
        ]
        assert config["temperature"] == 0.9
        assert config["top_k"] == 40

    async def test_create_batch_model_prefix(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test that model name gets models/ prefix."""
        httpx_mock.add_response(
            method="POST",
            url=BATCH_GENERATE_URL,
            json=gemini_batch_response(batch_name="batches/abc123"),
        )

        await adapter.create_batch(["Test"], mock_model)

        # Verify models/ prefix in URL
        request = httpx_mock.get_request()
        assert "models/gemini-2.5-flash:batchGenerateContent" in str(request.url)


class TestGetStatus:
    """Test get_status method."""

    async def test_get_status_in_progress(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting status for an in-progress batch."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_RUNNING",
                total=100,
                succeeded=50,
                pending=50,
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
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_SUCCEEDED",
                total=100,
                succeeded=100,
                pending=0,
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
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_FAILED",
                total=100,
                succeeded=0,
                failed=100,
                pending=0,
            ),
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.FAILED, total=100, completed=0, failed=100
        )

    @pytest.mark.parametrize(
        "gemini_state,expected_status",
        [
            ("BATCH_STATE_PENDING", BatchStatus.PENDING),
            ("BATCH_STATE_RUNNING", BatchStatus.IN_PROGRESS),
            ("BATCH_STATE_SUCCEEDED", BatchStatus.COMPLETED),
            ("BATCH_STATE_FAILED", BatchStatus.FAILED),
            ("BATCH_STATE_CANCELLED", BatchStatus.CANCELLED),
            ("BATCH_STATE_EXPIRED", BatchStatus.EXPIRED),
        ],
    )
    async def test_status_mapping(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock, gemini_state, expected_status
    ):
        """Test all Gemini state values map correctly."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state=gemini_state,
                total=10,
                succeeded=5,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result.status == expected_status


class TestGetResults:
    """Test get_results method."""

    async def test_get_results_success(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results from a completed batch."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_SUCCEEDED",
                done=True,
                inlined_responses=[
                    gemini_inline_result("0", content="Hello!"),
                    gemini_inline_result(
                        "1", content="I'm fine!", prompt_tokens=12, completion_tokens=8
                    ),
                ],
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(content="Hello!"),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            BatchItem(
                custom_id="1",
                success=True,
                content=AIMessage(content="I'm fine!"),
                usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            ),
        ]

    async def test_get_results_with_errors(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results with some failed requests."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_SUCCEEDED",
                done=True,
                inlined_responses=[
                    gemini_inline_result("0", content="Success!"),
                    gemini_error_result("1", error_message="Rate limit exceeded"),
                ],
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(content="Success!"),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            BatchItem(
                custom_id="1",
                success=False,
                error={"code": 13, "message": "Rate limit exceeded"},
            ),
        ]

    async def test_get_results_with_tool_calls(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock, freeze_uuid
    ):
        """Test getting results that contain tool calls."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_SUCCEEDED",
                done=True,
                inlined_responses=[
                    gemini_tool_call_inline_result(
                        "0",
                        tool_name="get_weather",
                        tool_args={"location": "NYC"},
                    ),
                ],
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchItem(
                custom_id="0",
                success=True,
                content=AIMessage(
                    content=[
                        {"text": "I'll check the weather."},
                        {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                    ],
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                            "id": freeze_uuid,
                            "type": "tool_call",
                        }
                    ],
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            ),
        ]

    async def test_get_results_no_responses(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results when batch has no inline responses."""
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_RUNNING",
                done=False,
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
            json=gemini_list_batches_response(
                [
                    gemini_batch_response(batch_name="batches/abc123"),
                    gemini_batch_response(
                        batch_name="batches/def456",
                        created_at="2024-01-15T12:01:40.000000Z",
                    ),
                ]
            ),
        )

        results = await adapter.list_batches(limit=10)

        assert results == [
            ProviderJob(
                id="batches/abc123",
                provider=Provider.GOOGLE,
                created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            ProviderJob(
                id="batches/def456",
                provider=Provider.GOOGLE,
                created_at=datetime(2024, 1, 15, 12, 1, 40, tzinfo=timezone.utc),
            ),
        ]

        # Verify pageSize was passed
        request = httpx_mock.get_request()
        assert "pageSize=10" in str(request.url)


class TestCancel:
    """Test cancel method."""

    async def test_cancel_batch(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test cancelling a batch waits until terminal state."""
        # Mock cancel request
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/batches/abc123:cancel",
            json={},
        )

        # Mock get_status calls - first running, then cancelled
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_RUNNING",
                total=100,
                succeeded=50,
                pending=50,
            ),
        )
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_CANCELLED",
                total=100,
                succeeded=50,
                failed=0,
                pending=0,
            ),
        )

        result = await adapter.cancel(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=100,
            completed=50,
            failed=0,
        )

        # Verify cancel was called then status polled twice
        requests = httpx_mock.get_requests()
        assert str(requests[0].url) == f"{BASE_URL}/batches/abc123:cancel"
        assert str(requests[1].url) == f"{BASE_URL}/batches/abc123"
        assert str(requests[2].url) == f"{BASE_URL}/batches/abc123"

    async def test_cancel_completes_to_cancelled(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test that status transitions from running to cancelled."""
        # First call: running
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_RUNNING",
                total=100,
                succeeded=50,
                pending=50,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=100, completed=50, failed=0
        )

        # Second call: cancelled
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/batches/abc123",
            json=gemini_batch_status_response(
                batch_name="batches/abc123",
                state="BATCH_STATE_CANCELLED",
                total=100,
                succeeded=50,
                pending=0,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED, total=100, completed=50, failed=0
        )


class TestAdapterInit:
    """Test adapter initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        settings = LangasyncSettings(google_api_key="test-google-key")
        adapter = GeminiProviderJobAdapter(settings)
        assert adapter.api_key == "test-google-key"
        assert adapter.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        settings = LangasyncSettings(
            google_api_key="test-google-key",
            google_base_url="https://custom.api.com/v1beta",
        )
        adapter = GeminiProviderJobAdapter(settings)
        assert adapter.base_url == "https://custom.api.com/v1beta"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        settings = LangasyncSettings(google_api_key=None)
        with pytest.raises(Exception, match="Google API key required"):
            GeminiProviderJobAdapter(settings)

    def test_init_from_settings(self):
        """Test initialization from settings object."""
        settings = LangasyncSettings(google_api_key="key-from-settings")
        adapter = GeminiProviderJobAdapter(settings)
        assert adapter.api_key == "key-from-settings"
