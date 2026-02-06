"""Unit tests for OpenAIProviderJobAdapter with mocked httpx calls."""

import json
import re
from datetime import datetime

import pytest
from pytest_httpx import HTTPXMock

from langchain_core.messages import AIMessage

from langasync.providers.openai import OpenAIProviderJobAdapter
from langasync.providers.interface import (
    ProviderJob,
    BatchResponse,
    BatchStatus,
    BatchStatusInfo,
    Provider,
)
from tests.fixtures.openai_responses import (
    openai_batch_response,
    openai_batch_status_response,
    openai_file_upload_response,
    openai_output_line,
    openai_error_output_line,
    openai_tool_call_output_line,
    openai_results_jsonl,
)


# URL patterns for matching (ignores query parameters)
FILES_URL = re.compile(r"https://api\.openai\.com/v1/files.*")
BATCHES_URL = re.compile(r"https://api\.openai\.com/v1/batches(\?.*)?$")
BATCH_ID_URL = re.compile(r"https://api\.openai\.com/v1/batches/batch_abc123.*")


@pytest.fixture
def adapter():
    """Create OpenAI adapter with test API key."""
    return OpenAIProviderJobAdapter(api_key="test-api-key")


@pytest.fixture
def mock_model():
    """Create a real ChatOpenAI model for testing."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4", temperature=0.7, api_key="test-key")


@pytest.fixture
def sample_batch_job():
    """Create a sample ProviderJob for testing."""
    return ProviderJob(
        id="batch_abc123",
        provider=Provider.OPENAI,
        created_at=datetime(2024, 1, 15, 12, 0, 0),
        metadata={"input_file_id": "file-xyz789"},
    )


class TestCreateBatch:
    """Test create_batch method."""

    async def test_create_batch_success(self, adapter, mock_model, httpx_mock: HTTPXMock):
        """Test successful batch creation."""
        # Mock file upload
        httpx_mock.add_response(
            method="POST",
            url=FILES_URL,
            json=openai_file_upload_response(file_id="file-xyz789"),
        )

        # Mock batch creation
        httpx_mock.add_response(
            method="POST",
            url=BATCHES_URL,
            json=openai_batch_response(
                batch_id="batch_abc123",
                status="validating",
                input_file_id="file-xyz789",
            ),
        )

        inputs = ["Hello, world!", "How are you?"]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id="batch_abc123",
            provider=Provider.OPENAI,
            created_at=datetime.fromtimestamp(1705320000),
            metadata={"input_file_id": "file-xyz789"},
        )

        # Verify batch request content
        requests = httpx_mock.get_requests()
        assert len(requests) == 2
        assert requests[0].url.path == "/v1/files"
        assert requests[1].url.path == "/v1/batches"

        content = requests[0].content.decode("utf-8")
        for i, msg in enumerate(["Hello, world!", "How are you?"]):
            expected = json.dumps(
                {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "messages": [{"role": "user", "content": msg}],
                    },
                }
            )
            assert expected in content

    async def test_create_batch_with_model_bindings(
        self, adapter, mock_model, httpx_mock: HTTPXMock
    ):
        """Test batch creation with model bindings (tools, etc.)."""
        httpx_mock.add_response(
            method="POST",
            url=FILES_URL,
            json=openai_file_upload_response(file_id="file-xyz789"),
        )
        httpx_mock.add_response(
            method="POST",
            url=BATCHES_URL,
            json=openai_batch_response(batch_id="batch_abc123"),
        )

        bindings = {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
        }
        result = await adapter.create_batch(["Test"], mock_model, model_bindings=bindings)

        assert result == ProviderJob(
            id="batch_abc123",
            provider=Provider.OPENAI,
            created_at=datetime.fromtimestamp(1705320000),
            metadata={"input_file_id": "file-xyz789"},
        )

        # Verify the batch request content
        content = httpx_mock.get_requests()[0].content.decode("utf-8")
        expected = json.dumps(
            {
                "custom_id": "0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                    "tool_choice": "auto",
                    "messages": [{"role": "user", "content": "Test"}],
                },
            }
        )
        assert expected in content


class TestGetStatus:
    """Test get_status method."""

    async def test_get_status_in_progress(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting status for an in-progress batch."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="in_progress",
                total=100,
                completed=50,
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
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="completed",
                total=100,
                completed=100,
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
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="failed",
                total=100,
                completed=0,
                failed=100,
            ),
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.FAILED, total=100, completed=0, failed=100
        )

    @pytest.mark.parametrize(
        "openai_status,expected_status",
        [
            ("validating", BatchStatus.VALIDATING),
            ("in_progress", BatchStatus.IN_PROGRESS),
            ("finalizing", BatchStatus.IN_PROGRESS),
            ("completed", BatchStatus.COMPLETED),
            ("failed", BatchStatus.FAILED),
            ("cancelled", BatchStatus.CANCELLED),
            ("cancelling", BatchStatus.IN_PROGRESS),
            ("expired", BatchStatus.EXPIRED),
        ],
    )
    async def test_status_mapping(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock, openai_status, expected_status
    ):
        """Test all OpenAI status values map correctly."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status=openai_status,
                total=10,
                completed=5,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result.status == expected_status


class TestGetResults:
    """Test get_results method."""

    async def test_get_results_success(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test getting results from a completed batch."""
        # Mock batch status
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_response(
                batch_id="batch_abc123",
                status="completed",
                output_file_id="file-output123",
            ),
        )

        # Mock output file download
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/files/file-output123/content",
            text=openai_results_jsonl(
                [
                    openai_output_line("0", content="Hello!"),
                    openai_output_line(
                        "1", content="I'm fine!", prompt_tokens=12, completion_tokens=8
                    ),
                ]
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchResponse(
                custom_id="0",
                success=True,
                content=AIMessage(content="Hello!"),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            BatchResponse(
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
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_response(
                batch_id="batch_abc123",
                status="completed",
                output_file_id="file-output123",
                error_file_id="file-error123",
            ),
        )

        # Output file with successful response
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/files/file-output123/content",
            text=openai_results_jsonl(
                [
                    openai_output_line("0", content="Success!"),
                ]
            ),
        )

        # Error file with failed response
        error_content = json.dumps(
            {
                "custom_id": "1",
                "error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"},
            }
        )
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/files/file-error123/content",
            text=error_content,
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchResponse(
                custom_id="0",
                success=True,
                content=AIMessage(content="Success!"),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            BatchResponse(
                custom_id="1",
                success=False,
                error={"message": "Rate limit exceeded", "code": "rate_limit_exceeded"},
            ),
        ]

    async def test_get_results_with_tool_calls(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test getting results that contain tool calls."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_response(
                batch_id="batch_abc123",
                status="completed",
                output_file_id="file-output123",
            ),
        )

        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/files/file-output123/content",
            text=openai_results_jsonl(
                [
                    openai_tool_call_output_line(
                        "0",
                        tool_name="get_weather",
                        tool_id="call_123",
                        arguments='{"location": "NYC"}',
                    ),
                ]
            ),
        )

        results = await adapter.get_results(sample_batch_job)

        assert results == [
            BatchResponse(
                custom_id="0",
                success=True,
                content=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                            "id": "call_123",
                            "type": "tool_call",
                        }
                    ],
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            ),
        ]


class TestListBatches:
    """Test list_batches method."""

    async def test_list_batches(self, adapter, httpx_mock: HTTPXMock):
        """Test listing batches."""
        httpx_mock.add_response(
            method="GET",
            url=BATCHES_URL,
            json={
                "data": [
                    openai_batch_response(
                        batch_id="batch_abc123",
                        input_file_id="file-xyz789",
                    ),
                    openai_batch_response(
                        batch_id="batch_def456",
                        input_file_id="file-uvw012",
                        created_at=1705320100,
                    ),
                ],
            },
        )

        results = await adapter.list_batches(limit=10)

        assert results == [
            ProviderJob(
                id="batch_abc123",
                provider=Provider.OPENAI,
                created_at=datetime.fromtimestamp(1705320000),
                metadata={"input_file_id": "file-xyz789"},
            ),
            ProviderJob(
                id="batch_def456",
                provider=Provider.OPENAI,
                created_at=datetime.fromtimestamp(1705320100),
                metadata={"input_file_id": "file-uvw012"},
            ),
        ]

        # Verify limit was passed
        request = httpx_mock.get_request()
        assert "limit=10" in str(request.url)


class TestCancel:
    """Test cancel method."""

    async def test_cancel_batch(self, adapter, sample_batch_job, httpx_mock: HTTPXMock):
        """Test cancelling a batch waits until cancelled."""
        # Mock cancel request
        httpx_mock.add_response(
            method="POST",
            url="https://api.openai.com/v1/batches/batch_abc123/cancel",
            json=openai_batch_response(
                batch_id="batch_abc123",
                status="cancelling",
            ),
        )

        # Mock get_status calls - first cancelling, then cancelled
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="cancelling",
                total=100,
                completed=50,
            ),
        )
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="cancelled",
                total=100,
                completed=50,
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
        assert requests[0].url.path == "/v1/batches/batch_abc123/cancel"
        assert requests[1].url.path == "/v1/batches/batch_abc123"
        assert requests[2].url.path == "/v1/batches/batch_abc123"

    async def test_cancel_completes_to_cancelled(
        self, adapter, sample_batch_job, httpx_mock: HTTPXMock
    ):
        """Test that status transitions from cancelling to cancelled."""
        # First call: cancelling
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="cancelling",
                total=100,
                completed=50,
            ),
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=100, completed=50, failed=0
        )

        # Second call: cancelled
        httpx_mock.add_response(
            method="GET",
            url="https://api.openai.com/v1/batches/batch_abc123",
            json=openai_batch_status_response(
                batch_id="batch_abc123",
                status="cancelled",
                total=100,
                completed=50,
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
        adapter = OpenAIProviderJobAdapter(api_key="sk-test123")
        assert adapter.api_key == "sk-test123"
        assert adapter.base_url == "https://api.openai.com/v1"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        adapter = OpenAIProviderJobAdapter(
            api_key="sk-test123", base_url="https://custom.api.com/v1/"
        )
        assert adapter.base_url == "https://custom.api.com/v1"  # Trailing slash removed

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test initialization without API key raises error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProviderJobAdapter()

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        adapter = OpenAIProviderJobAdapter()
        assert adapter.api_key == "sk-from-env"
