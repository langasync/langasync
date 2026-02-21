"""AWS Bedrock Batch Inference adapter."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import aioboto3
from langchain_anthropic.chat_models import _format_messages
from langchain_anthropic.output_parsers import extract_tool_calls
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue

from langasync.exceptions import ApiTimeoutError, AuthenticationError, provider_error_handling
from langasync.settings import LangasyncSettings
from langasync.utils import generate_uuid

from langasync.providers.interface import (
    FINISHED_STATUSES,
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
    Provider,
)

logger = logging.getLogger(__name__)


# --- Message Conversion (Bedrock Claude uses Anthropic Messages format) ---


def _convert_to_bedrock_messages(
    inp: LanguageModelInput,
) -> tuple[str | list[dict] | None, list[dict]]:
    """Convert LangChain input to Bedrock/Anthropic messages format."""
    if isinstance(inp, PromptValue):
        inp = inp.to_messages()
    if isinstance(inp, str):
        inp = [HumanMessage(content=inp)]
    if isinstance(inp, BaseMessage):
        inp = [inp]
    system, messages = _format_messages(inp)  # type: ignore[arg-type]
    return system, messages


def _to_bedrock_record(inp: LanguageModelInput, model_config: dict, record_id: str) -> dict:
    """Convert LanguageModelInput to Bedrock batch record format."""
    system, messages = _convert_to_bedrock_messages(inp)
    model_input = {**model_config, "messages": messages}
    if system:
        model_input["system"] = system
    return {"recordId": record_id, "modelInput": model_input}


def _map_bedrock_status(status: str) -> BatchStatus:
    """Map Bedrock job status to our BatchStatus enum."""
    mapping = {
        "Submitted": BatchStatus.PENDING,
        "Validating": BatchStatus.VALIDATING,
        "Scheduled": BatchStatus.PENDING,
        "InProgress": BatchStatus.IN_PROGRESS,
        "Completed": BatchStatus.COMPLETED,
        "PartiallyCompleted": BatchStatus.COMPLETED,
        "Failed": BatchStatus.FAILED,
        "Stopping": BatchStatus.IN_PROGRESS,
        "Stopped": BatchStatus.CANCELLED,
        "Expired": BatchStatus.EXPIRED,
    }
    return mapping.get(status, BatchStatus.PENDING)


class BedrockProviderJobAdapter(ProviderJobAdapterInterface):
    """AWS Bedrock Batch Inference adapter.

    Uses Bedrock's CreateModelInvocationJob API with S3 for input/output.
    Supports Claude models on Bedrock using the Anthropic Messages format.

    AWS credentials are resolved by boto3's standard credential chain
    (env vars, ~/.aws/credentials, instance profiles, SSO).
    """

    def __init__(self, settings: LangasyncSettings):
        self.region = settings.aws_region

        s3_bucket = settings.bedrock_s3_bucket
        if not s3_bucket:
            raise AuthenticationError("S3 bucket required. Set BEDROCK_S3_BUCKET.")
        self.s3_bucket: str = s3_bucket

        role_arn = settings.bedrock_role_arn
        if not role_arn:
            raise AuthenticationError("IAM role ARN required. Set BEDROCK_ROLE_ARN.")
        self.role_arn: str = role_arn

        # Only pass explicit credentials if set; otherwise boto3 uses its credential chain
        session_kwargs: dict[str, Any] = {"region_name": self.region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            if settings.aws_session_token:
                session_kwargs["aws_session_token"] = settings.aws_session_token

        self._session = aioboto3.Session(**session_kwargs)

        self._bedrock_kwargs: dict[str, Any] = {}
        if settings.bedrock_base_url:
            self._bedrock_kwargs["endpoint_url"] = settings.bedrock_base_url

    def _get_model_config(
        self, language_model: LanguageModelType, model_bindings: dict | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Extract Bedrock model ID and Anthropic-format config from LangChain model.

        Returns:
            Tuple of (model_id for Bedrock API, model_config dict for modelInput)
        """
        model = (
            getattr(language_model, "model_id", None)
            or getattr(language_model, "model_name", None)
            or getattr(language_model, "model", None)
        )
        if not model:
            raise ValueError(
                "Could not determine model name from language model. "
                "Ensure your model has a 'model', 'model_name', or 'model_id' attribute."
            )

        config: dict[str, Any] = {"anthropic_version": "bedrock-2023-05-31"}

        for param in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
            value = getattr(language_model, param, None)
            if value is not None:
                config[param] = value

        # max_tokens is required for Anthropic on Bedrock
        if "max_tokens" not in config:
            config["max_tokens"] = 1024

        config.update(getattr(language_model, "model_kwargs", {}))

        if model_bindings:
            config.update(model_bindings)

        return model, config

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
        model_bindings: dict | None = None,
    ) -> ProviderJob:
        """Create a new batch inference job with Bedrock."""
        model_id, model_config = self._get_model_config(language_model, model_bindings)

        # Build JSONL content
        records = [_to_bedrock_record(inp, model_config, str(i)) for i, inp in enumerate(inputs)]
        jsonl_content = "\n".join(json.dumps(r) for r in records)

        # Upload input to S3
        batch_uuid = generate_uuid()
        input_key = f"langasync/{batch_uuid}/input.jsonl"
        output_prefix = f"langasync/{batch_uuid}/output/"
        logger.debug(f"Uploading {len(inputs)} records to s3://{self.s3_bucket}/{input_key}")

        async with self._session.client("s3") as s3:
            await s3.put_object(
                Bucket=self.s3_bucket,
                Key=input_key,
                Body=jsonl_content.encode("utf-8"),
                ContentType="application/jsonl",
            )

        # Create batch inference job
        async with self._session.client("bedrock", **self._bedrock_kwargs) as bedrock:
            response = await bedrock.create_model_invocation_job(
                modelId=model_id,
                jobName=f"langasync-{batch_uuid}",
                roleArn=self.role_arn,
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3Uri": f"s3://{self.s3_bucket}/{input_key}",
                        "s3InputFormat": "JSONL",
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"s3://{self.s3_bucket}/{output_prefix}",
                    }
                },
            )

        job_arn = response["jobArn"]
        logger.info(f"Bedrock batch created: {job_arn}")

        return ProviderJob(
            id=job_arn,
            provider=Provider.BEDROCK,
            created_at=datetime.now(timezone.utc),
            metadata={
                "input_key": input_key,
                "output_prefix": output_prefix,
                "model_id": model_id,
                "total_records": len(inputs),
            },
        )

    async def _read_manifest(self, output_prefix: str) -> dict | None:
        """Read manifest.json.out from S3 output prefix for record counts."""
        manifest_key = f"{output_prefix}manifest.json.out"
        try:
            async with self._session.client("s3") as s3:
                response = await s3.get_object(Bucket=self.s3_bucket, Key=manifest_key)
                body = await response["Body"].read()
                return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    @provider_error_handling
    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Get the current status of a batch job."""
        job_id = batch_api_job.id.split("/")[-1] if "/" in batch_api_job.id else batch_api_job.id

        async with self._session.client("bedrock", **self._bedrock_kwargs) as bedrock:
            data = await bedrock.get_model_invocation_job(jobIdentifier=job_id)

        status = data.get("status", "Submitted")
        mapped_status = _map_bedrock_status(status)
        logger.info(f"Bedrock batch {batch_api_job.id}: status={status}")

        # Record counts come from manifest.json.out in the S3 output prefix.
        total = batch_api_job.metadata.get("total_records", 0)
        completed = 0
        failed = 0

        output_prefix = batch_api_job.metadata.get("output_prefix", "")
        if output_prefix:
            manifest = await self._read_manifest(output_prefix)
            if manifest:
                total = manifest.get("totalRecordCount", total)
                completed = manifest.get("successRecordCount", 0)
                failed = manifest.get("errorRecordCount", 0)

        return BatchStatusInfo(
            status=mapped_status,
            total=total,
            completed=completed,
            failed=failed,
        )

    @provider_error_handling
    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        """List recent batch jobs."""
        async with self._session.client("bedrock", **self._bedrock_kwargs) as bedrock:
            data = await bedrock.list_model_invocation_jobs(maxResults=limit)

        return [
            ProviderJob(
                id=job["jobArn"],
                provider=Provider.BEDROCK,
                created_at=(
                    job["submitTime"]
                    if isinstance(job["submitTime"], datetime)
                    else datetime.fromisoformat(job["submitTime"].replace("Z", "+00:00"))
                ),
                metadata={"model_id": job.get("modelId", "")},
            )
            for job in data.get("invocationJobSummaries", [])
        ]

    def _parse_result_line(self, data: dict) -> BatchItem:
        """Parse a single line from Bedrock output JSONL."""
        record_id = data.get("recordId", "")
        model_output = data.get("modelOutput")
        error = data.get("error")

        if error:
            return BatchItem(
                custom_id=record_id,
                success=False,
                error=error,
            )

        if model_output:
            content_blocks = model_output.get("content", [])

            # Match LangChain ChatAnthropic behavior (same as anthropic.py)
            if (
                len(content_blocks) == 1
                and content_blocks[0].get("type") == "text"
                and not content_blocks[0].get("citations")
            ):
                content = content_blocks[0].get("text", "")
                ai_message = AIMessage(content=content)
            elif any(b.get("type") == "tool_use" for b in content_blocks):
                tool_calls = extract_tool_calls(content_blocks)
                ai_message = AIMessage(content=content_blocks, tool_calls=tool_calls)
            else:
                ai_message = AIMessage(content=content_blocks)

            return BatchItem(
                custom_id=record_id,
                success=True,
                content=ai_message,
                usage=model_output.get("usage"),
            )

        return BatchItem(
            custom_id=record_id,
            success=False,
            error={"message": "No output or error in response"},
        )

    @provider_error_handling
    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        """Get results from a completed batch job."""
        output_prefix = batch_api_job.metadata.get("output_prefix", "")
        if not output_prefix:
            return []

        logger.debug(f"Listing output files for batch {batch_api_job.id}")

        async with self._session.client("s3") as s3:
            # List output files in S3
            list_response = await s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=output_prefix,
            )

            output_keys = [
                obj["Key"]
                for obj in list_response.get("Contents", [])
                if obj["Key"].endswith(".jsonl.out")
            ]

            if not output_keys:
                return []

            # Download and parse all output files
            results = []
            for key in output_keys:
                response = await s3.get_object(Bucket=self.s3_bucket, Key=key)
                body = await response["Body"].read()
                content = body.decode("utf-8")
                for line in content.strip().split("\n"):
                    if line:
                        result = self._parse_result_line(json.loads(line))
                        results.append(result)

        results.sort(key=lambda r: int(r.custom_id) if r.custom_id.isdigit() else 0)
        return results

    @provider_error_handling
    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        """Cancel a batch job and wait until cancellation completes."""
        job_id = batch_api_job.id.split("/")[-1] if "/" in batch_api_job.id else batch_api_job.id

        async with self._session.client("bedrock", **self._bedrock_kwargs) as bedrock:
            await bedrock.stop_model_invocation_job(jobIdentifier=job_id)

        # Poll until batch reaches a terminal state
        cancel_timeout_seconds = 60
        for _ in range(cancel_timeout_seconds):
            status_info = await self.get_status(batch_api_job)
            if status_info.status in FINISHED_STATUSES:
                return BatchStatusInfo(
                    status=BatchStatus.CANCELLED,
                    total=status_info.total,
                    completed=status_info.completed,
                    failed=status_info.failed,
                )
            await asyncio.sleep(1)
        raise ApiTimeoutError(f"Cancel timed out after {cancel_timeout_seconds} seconds")
