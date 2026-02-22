"""Unit tests for BedrockProviderJobAdapter with mocked aioboto3."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from freezegun import freeze_time

import pytest

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from langasync.exceptions import BatchProviderApiError, UnsupportedProviderError
from langasync.providers.bedrock import (
    BEDROCK_MIN_BATCH_SIZE,
    BedrockProviderJobAdapter,
    get_provider,
    get_provider_from_model,
)
from langasync.providers.bedrock.model_providers import _get_provider_str
from langasync.providers.interface import (
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    Provider,
)
from langasync.settings import LangasyncSettings
from tests.fixtures.bedrock_responses import (
    bedrock_create_job_response,
    bedrock_job_status_response,
    bedrock_list_jobs_response,
    bedrock_manifest,
    bedrock_output_line,
    bedrock_error_output_line,
    bedrock_model_output_error_line,
    bedrock_tool_use_output_line,
    bedrock_results_jsonl,
)

JOB_ARN = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id"
FROZEN_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
TOTAL_RECORDS = 100
BATCH_UUID = "test-batch-uuid"


@pytest.fixture
def bedrock_settings(tmp_path):
    """Create settings with AWS/Bedrock credentials."""
    return LangasyncSettings(
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_region="us-east-1",
        bedrock_s3_bucket="test-bucket",
        bedrock_role_arn="arn:aws:iam::123456789012:role/TestRole",
        base_storage_path=str(tmp_path),
    )


@pytest.fixture
def adapter(bedrock_settings):
    """Create Bedrock adapter with mocked aioboto3 session."""
    mock_s3 = AsyncMock()
    mock_bedrock = AsyncMock()

    def client_factory(service, **_kwargs):
        cm = AsyncMock()
        if service == "s3":
            cm.__aenter__.return_value = mock_s3
        elif service == "bedrock":
            cm.__aenter__.return_value = mock_bedrock
        return cm

    with patch("langasync.providers.bedrock.core.aioboto3") as mock_aioboto3:
        mock_session = MagicMock()
        mock_session.client.side_effect = client_factory
        mock_aioboto3.Session.return_value = mock_session
        adapter = BedrockProviderJobAdapter(bedrock_settings)

    # Expose mocks for test assertions
    adapter._mock_bedrock = mock_bedrock
    adapter._mock_s3 = mock_s3
    return adapter


@pytest.fixture
def mock_model():
    """Create a mock model with Bedrock-style attributes."""
    return SimpleNamespace(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_name=None,
        model=None,
        max_tokens=1024,
        temperature=0.7,
        top_p=None,
        top_k=None,
        stop_sequences=None,
        model_kwargs={},
    )


@pytest.fixture
def sample_batch_job():
    """Create a sample ProviderJob for testing."""
    return ProviderJob(
        id=JOB_ARN,
        provider=Provider.BEDROCK,
        created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        metadata={
            "input_key": "langasync/test-uuid/input.jsonl",
            "output_prefix": "langasync/test-uuid/output/",
            "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock_provider": "anthropic",
            "total_records": TOTAL_RECORDS,
        },
    )


def _s3_get_object_response(content: str) -> dict:
    """Create a mock S3 get_object response with an async-readable Body."""
    body = AsyncMock()
    body.read.return_value = content.encode("utf-8")
    return {"Body": body}


@freeze_time(FROZEN_NOW)
class TestCreateBatch:
    """Test create_batch method."""

    @pytest.fixture(autouse=True)
    def _freeze_uuid(self, freeze_uuid):
        self.batch_uuid = freeze_uuid

    async def test_create_batch_rejects_too_few_inputs(self, adapter, mock_model):
        """Test that create_batch raises when inputs < BEDROCK_MIN_BATCH_SIZE."""
        with pytest.raises(
            BatchProviderApiError, match=f"at least {BEDROCK_MIN_BATCH_SIZE} records"
        ):
            await adapter.create_batch(["Hello"], mock_model)

    async def test_create_batch_success(self, adapter, mock_model):
        """Test successful batch creation (S3 upload + Bedrock job create)."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{self.batch_uuid}/input.jsonl",
                "output_prefix": f"langasync/{self.batch_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify S3 upload content
        put_call = adapter._mock_s3.put_object.call_args
        assert put_call.kwargs["Bucket"] == "test-bucket"
        assert put_call.kwargs["ContentType"] == "application/jsonl"
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        assert len(lines) == BEDROCK_MIN_BATCH_SIZE

        record_0 = json.loads(lines[0])
        assert record_0 == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }

        # Verify Bedrock create request
        create_call = adapter._mock_bedrock.create_model_invocation_job.call_args
        assert create_call.kwargs["modelId"] == "us.anthropic.claude-3-sonnet-20240229-v1:0"
        assert create_call.kwargs["roleArn"] == "arn:aws:iam::123456789012:role/TestRole"
        assert (
            create_call.kwargs["inputDataConfig"]["s3InputDataConfig"]["s3InputFormat"] == "JSONL"
        )

    async def test_create_batch_with_system_message(self, adapter, mock_model):
        """Test batch creation with system message."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        msg_with_system = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
        ]
        inputs = [msg_with_system] + [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE - 1)]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{self.batch_uuid}/input.jsonl",
                "output_prefix": f"langasync/{self.batch_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify system message is extracted to top-level system field in first record
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        }

    async def test_create_batch_with_image_input(self, adapter, mock_model):
        """Test batch creation with image content in messages."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        msg_with_image = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "url": "https://example.com/cat.jpg"},
                ]
            ),
        ]
        inputs = [msg_with_image] + [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE - 1)]
        result = await adapter.create_batch(inputs, mock_model)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{self.batch_uuid}/input.jsonl",
                "output_prefix": f"langasync/{self.batch_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify batch request content — _format_messages converts to Anthropic format
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
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

    async def test_create_batch_with_model_bindings(self, adapter, mock_model):
        """Test batch creation with model bindings (tools, etc.)."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        # OpenAI format — what ChatBedrockConverse.bind_tools() produces
        bindings = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        }
        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, mock_model, model_bindings=bindings)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{self.batch_uuid}/input.jsonl",
                "output_prefix": f"langasync/{self.batch_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify tools converted to Anthropic format in the record
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": {"type": "auto"},
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }

    @freeze_time("2024-01-15 12:00:00", tz_offset=0)
    async def test_create_batch_with_anthropic_format_tools(self, adapter, mock_model):
        """ChatBedrock.bind_tools() produces Anthropic-format tools — passed through unchanged."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        bindings = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto"},
        }
        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        await adapter.create_batch(inputs, mock_model, model_bindings=bindings)

        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])

        # Anthropic-format tools should pass through unchanged
        assert record["modelInput"]["tools"] == [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]


class TestGetStatus:
    """Test get_status method."""

    async def test_get_status_in_progress_with_manifest(self, adapter, sample_batch_job):
        """Test getting status reads manifest.json.out for record counts."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="InProgress"
        )
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/manifest.json.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            json.dumps(
                bedrock_manifest(
                    total_record_count=100,
                    success_record_count=50,
                    error_record_count=2,
                )
            )
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=100, completed=50, failed=2
        )

    async def test_get_status_in_progress_no_manifest(self, adapter, sample_batch_job):
        """Test getting status falls back to metadata when manifest not yet available."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="InProgress"
        )
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=TOTAL_RECORDS, completed=0, failed=0
        )

    async def test_get_status_completed(self, adapter, sample_batch_job):
        """Test getting status for a completed batch."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="Completed"
        )
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/manifest.json.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            json.dumps(
                bedrock_manifest(
                    total_record_count=100,
                    success_record_count=100,
                    error_record_count=0,
                )
            )
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.COMPLETED, total=100, completed=100, failed=0
        )

    async def test_get_status_partially_completed(self, adapter, sample_batch_job):
        """Test getting status for a partially completed batch."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="PartiallyCompleted"
        )
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/manifest.json.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            json.dumps(
                bedrock_manifest(
                    total_record_count=100,
                    success_record_count=80,
                    error_record_count=20,
                )
            )
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.COMPLETED, total=100, completed=80, failed=20
        )

    async def test_get_status_failed(self, adapter, sample_batch_job):
        """Test getting status for a failed batch."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="Failed",
            message="Model invocation failed",
        )
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/manifest.json.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            json.dumps(
                bedrock_manifest(
                    total_record_count=100,
                    success_record_count=0,
                    error_record_count=100,
                )
            )
        )

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.FAILED, total=100, completed=0, failed=100
        )

    async def test_get_status_pending_no_manifest(self, adapter, sample_batch_job):
        """Test pending status with no manifest yet available."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="Submitted"
        )
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        result = await adapter.get_status(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.PENDING, total=TOTAL_RECORDS, completed=0, failed=0
        )

    @pytest.mark.parametrize(
        "bedrock_status,expected_status",
        [
            ("Submitted", BatchStatus.PENDING),
            ("Validating", BatchStatus.VALIDATING),
            ("Scheduled", BatchStatus.PENDING),
            ("InProgress", BatchStatus.IN_PROGRESS),
            ("Completed", BatchStatus.COMPLETED),
            ("PartiallyCompleted", BatchStatus.COMPLETED),
            ("Failed", BatchStatus.FAILED),
            ("Stopping", BatchStatus.IN_PROGRESS),
            ("Stopped", BatchStatus.CANCELLED),
            ("Expired", BatchStatus.EXPIRED),
        ],
    )
    async def test_status_mapping(
        self,
        adapter,
        sample_batch_job,
        bedrock_status,
        expected_status,
    ):
        """Test all Bedrock status values map correctly."""
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status=bedrock_status
        )
        # Manifest may or may not exist — not relevant to status mapping
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        result = await adapter.get_status(sample_batch_job)
        assert result.status == expected_status


class TestGetResults:
    """Test get_results method."""

    async def test_get_results_success(self, adapter, sample_batch_job):
        """Test getting results from a completed batch."""
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/input.jsonl.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            bedrock_results_jsonl(
                [
                    bedrock_output_line("0", content="Hello!"),
                    bedrock_output_line("1", content="I'm fine!", input_tokens=12, output_tokens=8),
                ]
            )
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

    async def test_get_results_with_errors(self, adapter, sample_batch_job):
        """Test getting results with some failed requests."""
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/input.jsonl.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            bedrock_results_jsonl(
                [
                    bedrock_output_line("0", content="Success!"),
                    bedrock_error_output_line(
                        "1",
                        error_code="ThrottlingException",
                        error_message="Too many requests",
                    ),
                ]
            )
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
                error={
                    "errorCode": "ThrottlingException",
                    "errorMessage": "Too many requests",
                },
            ),
        ]

    async def test_get_results_with_model_output_errors(self, adapter, sample_batch_job):
        """Test errors embedded in modelOutput (input validation failures)."""
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/input.jsonl.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            bedrock_results_jsonl(
                [
                    bedrock_output_line("0", content="Success!"),
                    bedrock_model_output_error_line(
                        "1",
                        error_code=400,
                        error_message="messages.0.content.1.image.source.base64.data: URL sources are not supported",
                    ),
                ]
            )
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
                error={
                    "errorCode": 400,
                    "errorMessage": "messages.0.content.1.image.source.base64.data: URL sources are not supported",
                    "expired": False,
                    "retryable": False,
                },
            ),
        ]

    async def test_get_results_with_tool_calls(self, adapter, sample_batch_job):
        """Test getting results that contain tool calls."""
        adapter._mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "langasync/test-uuid/output/test-job-id/input.jsonl.out"},
            ]
        }
        adapter._mock_s3.get_object.return_value = _s3_get_object_response(
            bedrock_results_jsonl(
                [
                    bedrock_tool_use_output_line(
                        "0",
                        tool_name="get_weather",
                        tool_id="call_123",
                        tool_input={"location": "NYC"},
                    ),
                ]
            )
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

    async def test_get_results_no_output_prefix(self, adapter):
        """Test getting results when no output_prefix in metadata."""
        job = ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            metadata={},
        )
        results = await adapter.get_results(job)
        assert results == []

    async def test_get_results_no_output_files(self, adapter, sample_batch_job):
        """Test getting results when no output files exist in S3."""
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        results = await adapter.get_results(sample_batch_job)
        assert results == []


class TestListBatches:
    """Test list_batches method."""

    async def test_list_batches(self, adapter):
        """Test listing batches."""
        adapter._mock_bedrock.list_model_invocation_jobs.return_value = bedrock_list_jobs_response(
            [
                {
                    "jobArn": JOB_ARN,
                    "status": "Completed",
                    "submitTime": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                },
                {
                    "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job2",
                    "status": "InProgress",
                    "submitTime": datetime(2024, 1, 15, 12, 1, 40, tzinfo=timezone.utc),
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                },
            ]
        )

        results = await adapter.list_batches(limit=10)

        assert results == [
            ProviderJob(
                id=JOB_ARN,
                provider=Provider.BEDROCK,
                created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                metadata={"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"},
            ),
            ProviderJob(
                id="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job2",
                provider=Provider.BEDROCK,
                created_at=datetime(2024, 1, 15, 12, 1, 40, tzinfo=timezone.utc),
                metadata={"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"},
            ),
        ]

        # Verify limit was passed
        adapter._mock_bedrock.list_model_invocation_jobs.assert_called_once_with(maxResults=10)


class TestCancel:
    """Test cancel method."""

    async def test_cancel_batch(self, adapter, sample_batch_job):
        """Test cancelling a batch waits until terminal state."""
        adapter._mock_bedrock.stop_model_invocation_job.return_value = {}

        # First call returns Stopping, second returns Stopped
        adapter._mock_bedrock.get_model_invocation_job.side_effect = [
            bedrock_job_status_response(status="Stopping"),
            bedrock_job_status_response(status="Stopped"),
        ]
        # Manifest may not exist for cancelled jobs
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        result = await adapter.cancel(sample_batch_job)

        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=TOTAL_RECORDS,
            completed=0,
            failed=0,
        )

        # Verify stop was called
        adapter._mock_bedrock.stop_model_invocation_job.assert_called_once_with(
            jobIdentifier=JOB_ARN
        )
        # Verify status was polled twice
        assert adapter._mock_bedrock.get_model_invocation_job.call_count == 2

    async def test_cancel_completes_to_cancelled(self, adapter, sample_batch_job):
        """Test that status transitions from Stopping to Stopped via get_status."""
        # First call: Stopping (maps to IN_PROGRESS)
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="Stopping"
        )
        adapter._mock_s3.list_objects_v2.return_value = {"Contents": []}

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS, total=TOTAL_RECORDS, completed=0, failed=0
        )

        # Second call: Stopped (maps to CANCELLED)
        adapter._mock_bedrock.get_model_invocation_job.return_value = bedrock_job_status_response(
            status="Stopped"
        )

        result = await adapter.get_status(sample_batch_job)
        assert result == BatchStatusInfo(
            status=BatchStatus.CANCELLED, total=TOTAL_RECORDS, completed=0, failed=0
        )


class TestProviderDetection:
    """Test _get_provider_str extracts provider from Bedrock model IDs."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("anthropic.claude-3-sonnet-20240229-v1:0", "anthropic"),
            ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic"),
            ("eu.anthropic.claude-sonnet-4-20250514-v1:0", "anthropic"),
            ("meta.llama3-70b-instruct-v1:0", "meta"),
            ("us.meta.llama3-2-90b-instruct-v1:0", "meta"),
            ("mistral.mistral-7b-instruct-v0:2", "mistral"),
            ("amazon.titan-text-express-v1", "amazon"),
            ("deepseek.deepseek-r1-v1:0", "deepseek"),
        ],
    )
    def test_get_provider(self, model_id: str, expected: str) -> None:
        assert _get_provider_str(model_id) == expected


class TestUnsupportedProvider:
    """Test that unsupported Bedrock providers raise UnsupportedProviderError."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta.llama3-70b-instruct-v1:0",
            "us.meta.llama3-2-90b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "deepseek.deepseek-r1-v1:0",
            "amazon.titan-text-express-v1",
        ],
    )
    def test_get_provider_rejects_unsupported(self, model_id: str) -> None:
        provider_str = _get_provider_str(model_id)
        with pytest.raises(UnsupportedProviderError):
            get_provider(provider_str, model_id, "us")

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta.llama3-70b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "deepseek.deepseek-r1-v1:0",
            "amazon.titan-text-express-v1",
        ],
    )
    def test_get_provider_from_model_rejects_unsupported(self, model_id: str) -> None:
        mock_model = SimpleNamespace(
            model_id=model_id,
            model_name=None,
            model=None,
            provider=None,
        )
        with pytest.raises(UnsupportedProviderError):
            get_provider_from_model(mock_model, "us")


class TestAdapterInit:
    """Test adapter initialization."""

    def test_init_with_explicit_credentials(self):
        """Test initialization passes explicit credentials to aioboto3 Session."""
        settings = LangasyncSettings(
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            aws_region="us-west-2",
            bedrock_s3_bucket="my-bucket",
            bedrock_role_arn="arn:aws:iam::123:role/R",
        )
        with patch("langasync.providers.bedrock.core.aioboto3") as mock_aioboto3:
            mock_aioboto3.Session.return_value = MagicMock()
            adapter = BedrockProviderJobAdapter(settings)

            assert adapter.region == "us-west-2"
            assert adapter.s3_bucket == "my-bucket"

            # Verify Session was created with explicit credentials
            mock_aioboto3.Session.assert_called_once_with(
                region_name="us-west-2",
                aws_access_key_id="AKID",
                aws_secret_access_key="secret",
            )

    def test_init_with_session_token(self):
        """Test initialization with temporary credentials (session token)."""
        settings = LangasyncSettings(
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            aws_session_token="token123",
            aws_region="us-east-1",
            bedrock_s3_bucket="my-bucket",
            bedrock_role_arn="arn:aws:iam::123:role/R",
        )
        with patch("langasync.providers.bedrock.core.aioboto3") as mock_aioboto3:
            mock_aioboto3.Session.return_value = MagicMock()
            BedrockProviderJobAdapter(settings)

            mock_aioboto3.Session.assert_called_once_with(
                region_name="us-east-1",
                aws_access_key_id="AKID",
                aws_secret_access_key="secret",
                aws_session_token="token123",
            )

    def test_init_with_custom_base_url(self):
        """Test initialization stores endpoint_url for bedrock client."""
        settings = LangasyncSettings(
            bedrock_s3_bucket="my-bucket",
            bedrock_role_arn="arn:aws:iam::123:role/R",
            bedrock_base_url="https://custom-bedrock.example.com",
        )
        with patch("langasync.providers.bedrock.core.aioboto3") as mock_aioboto3:
            mock_aioboto3.Session.return_value = MagicMock()
            adapter = BedrockProviderJobAdapter(settings)

            assert adapter._bedrock_kwargs == {"endpoint_url": "https://custom-bedrock.example.com"}

    def test_init_from_settings(self):
        """Test initialization from settings object."""
        settings = LangasyncSettings(
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            bedrock_s3_bucket="from-settings-bucket",
            bedrock_role_arn="arn:aws:iam::123:role/FromSettings",
        )
        with patch("langasync.providers.bedrock.core.aioboto3") as mock_aioboto3:
            mock_aioboto3.Session.return_value = MagicMock()
            adapter = BedrockProviderJobAdapter(settings)
            assert adapter.s3_bucket == "from-settings-bucket"
            assert adapter.role_arn == "arn:aws:iam::123:role/FromSettings"

    def test_init_without_s3_bucket_raises(self, monkeypatch):
        """Test initialization without S3 bucket raises error."""
        monkeypatch.delenv("BEDROCK_S3_BUCKET", raising=False)
        settings = LangasyncSettings(
            _env_file=None,
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            bedrock_role_arn="arn:aws:iam::123:role/R",
        )
        with (
            patch("langasync.providers.bedrock.core.aioboto3"),
            pytest.raises(Exception, match="S3 bucket required"),
        ):
            BedrockProviderJobAdapter(settings)

    def test_init_without_role_arn_raises(self, monkeypatch):
        """Test initialization without IAM role ARN raises error."""
        monkeypatch.delenv("BEDROCK_ROLE_ARN", raising=False)
        settings = LangasyncSettings(
            _env_file=None,
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            bedrock_s3_bucket="my-bucket",
        )
        with (
            patch("langasync.providers.bedrock.core.aioboto3"),
            pytest.raises(Exception, match="IAM role ARN required"),
        ):
            BedrockProviderJobAdapter(settings)


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str


class TestWithChatBedrock:
    """Tests using real ChatBedrock instances (no SimpleNamespace mocks)."""

    @pytest.fixture
    def chat_bedrock(self):
        return ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
        )

    @pytest.fixture
    def chat_bedrock_converse(self):
        return ChatBedrockConverse(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
        )

    def test_get_provider_from_chat_bedrock(self, chat_bedrock):
        """get_provider_from_model works with ChatBedrock."""
        provider = get_provider_from_model(chat_bedrock, "us")
        assert provider.model_id == "us.anthropic.claude-3-sonnet-20240229-v1:0"
        assert provider.bedrock_provider == "anthropic"

    def test_get_provider_from_chat_bedrock_converse(self, chat_bedrock_converse):
        """get_provider_from_model works with ChatBedrockConverse."""
        provider = get_provider_from_model(chat_bedrock_converse, "us")
        assert provider.model_id == "us.anthropic.claude-3-sonnet-20240229-v1:0"
        assert provider.bedrock_provider == "anthropic"

    def test_build_config_chat_bedrock_defaults(self, chat_bedrock):
        """ChatBedrock with no overrides gets anthropic_version and default max_tokens."""
        provider = get_provider_from_model(chat_bedrock, "us")
        config = provider.build_model_config(chat_bedrock)
        assert config == {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
        }

    def test_build_config_chat_bedrock_converse_defaults(self, chat_bedrock_converse):
        """ChatBedrockConverse with no overrides gets anthropic_version and default max_tokens."""
        provider = get_provider_from_model(chat_bedrock_converse, "us")
        config = provider.build_model_config(chat_bedrock_converse)
        assert config == {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
        }

    def test_chat_bedrock_tools_pass_through(self, chat_bedrock):
        """ChatBedrock.bind_tools() produces Anthropic format — passed through unchanged."""
        bound = chat_bedrock.bind_tools([GetWeather])
        provider = get_provider_from_model(bound, "us")
        config = provider.build_model_config(bound, model_bindings=bound.kwargs)

        assert config == {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "GetWeather",
                    "description": "Get the current weather in a given location.",
                    "input_schema": {
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "type": "object",
                    },
                }
            ],
        }

    def test_chat_bedrock_converse_tools_converted(self, chat_bedrock_converse):
        """ChatBedrockConverse.bind_tools() produces OpenAI format — converted to Anthropic."""
        bound = chat_bedrock_converse.bind_tools([GetWeather])
        provider = get_provider_from_model(bound, "us")
        config = provider.build_model_config(bound, model_bindings=bound.kwargs)

        assert config == {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "GetWeather",
                    "description": "Get the current weather in a given location.",
                    "input_schema": {
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "type": "object",
                    },
                }
            ],
        }

    @freeze_time(FROZEN_NOW)
    async def test_create_batch_with_chat_bedrock(self, adapter, chat_bedrock, freeze_uuid):
        """End-to-end create_batch with real ChatBedrock instance."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, chat_bedrock)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{freeze_uuid}/input.jsonl",
                "output_prefix": f"langasync/{freeze_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify record format
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        assert len(lines) == BEDROCK_MIN_BATCH_SIZE
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }

    @freeze_time(FROZEN_NOW)
    async def test_create_batch_with_chat_bedrock_converse(
        self, adapter, chat_bedrock_converse, freeze_uuid
    ):
        """End-to-end create_batch with real ChatBedrockConverse instance."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, chat_bedrock_converse)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{freeze_uuid}/input.jsonl",
                "output_prefix": f"langasync/{freeze_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        assert len(lines) == BEDROCK_MIN_BATCH_SIZE
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }

    @freeze_time(FROZEN_NOW)
    async def test_create_batch_with_chat_bedrock_converse_tools(
        self, adapter, chat_bedrock_converse, freeze_uuid
    ):
        """End-to-end create_batch with ChatBedrockConverse + bind_tools."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        bound = chat_bedrock_converse.bind_tools([GetWeather])
        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, bound, model_bindings=bound.kwargs)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{freeze_uuid}/input.jsonl",
                "output_prefix": f"langasync/{freeze_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify OpenAI-format tools are converted to Anthropic format in the JSONL
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "tools": [
                    {
                        "name": "GetWeather",
                        "description": "Get the current weather in a given location.",
                        "input_schema": {
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "type": "object",
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }

    @freeze_time(FROZEN_NOW)
    async def test_create_batch_with_chat_bedrock_tools(self, adapter, chat_bedrock, freeze_uuid):
        """End-to-end create_batch with ChatBedrock + bind_tools."""
        adapter._mock_s3.put_object.return_value = {}
        adapter._mock_bedrock.create_model_invocation_job.return_value = (
            bedrock_create_job_response(job_arn=JOB_ARN)
        )

        bound = chat_bedrock.bind_tools([GetWeather])
        inputs = [f"Message {i}" for i in range(BEDROCK_MIN_BATCH_SIZE)]
        result = await adapter.create_batch(inputs, bound, model_bindings=bound.kwargs)

        assert result == ProviderJob(
            id=JOB_ARN,
            provider=Provider.BEDROCK,
            created_at=FROZEN_NOW,
            metadata={
                "input_key": f"langasync/{freeze_uuid}/input.jsonl",
                "output_prefix": f"langasync/{freeze_uuid}/output/",
                "model_id": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock_provider": "anthropic",
                "total_records": BEDROCK_MIN_BATCH_SIZE,
            },
        )

        # Verify Anthropic-format tools pass through unchanged in the JSONL
        put_call = adapter._mock_s3.put_object.call_args
        lines = put_call.kwargs["Body"].decode().strip().split("\n")
        record = json.loads(lines[0])
        assert record == {
            "recordId": "0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "tools": [
                    {
                        "name": "GetWeather",
                        "description": "Get the current weather in a given location.",
                        "input_schema": {
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "type": "object",
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Message 0"}],
            },
        }
