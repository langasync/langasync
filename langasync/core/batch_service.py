from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable

from langasync.exceptions import (
    error_handling,
    LangAsyncError,
    UnsupportedProviderError,
    FailedLLMOutputError,
    FailedPostProcessingError,
    FailedPreProcessingError,
)

from langasync.providers.interface import (
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchStatusInfo,
    FINISHED_STATUSES,
    BatchStatus,
    BatchItem,
)
from langasync.core.batch_job_repository import BatchJob, BatchJobRepository
from langasync.providers import ADAPTER_REGISTRY, Provider


def _get_adapter_from_provider(provider: Provider) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a provider name."""
    adapter_cls = ADAPTER_REGISTRY.get(provider)
    if adapter_cls is None:
        raise UnsupportedProviderError(f"Unknown provider: {provider}")
    return adapter_cls()


def _get_provider_from_model(model: BaseLanguageModel | None) -> Provider:
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


def _get_adapter_from_model(model: BaseLanguageModel | None) -> ProviderJobAdapterInterface:
    """Get the appropriate batch API adapter for a model."""
    provider = _get_provider_from_model(model)
    return _get_adapter_from_provider(provider)


# NOTE: outputting dataclass because we want to allow error types
# and dont plan on serialising this
@dataclass
class ProcessedResults:
    """Result from get_results(), includes status and processed outputs."""

    job_id: str
    results: list[Any | LangAsyncError] | None
    status_info: BatchStatusInfo


class BatchJobService:
    def __init__(
        self,
        batch_api_job: ProviderJob,
        batch_api_adapter: ProviderJobAdapterInterface,
        postprocessing_chain: Runnable,
        repository: BatchJobRepository,
    ):
        self.batch_api_job = batch_api_job
        self.batch_api_adapter = batch_api_adapter
        self.postprocessing_chain = postprocessing_chain
        self.repository = repository

    @property
    def job_id(self) -> str:
        return self.batch_api_job.id

    @classmethod
    @error_handling
    async def create(
        cls,
        inputs: list[Any],
        model: BaseLanguageModel | None,
        preprocessing_chain: Runnable,
        postprocessing_chain: Runnable,
        repository: BatchJobRepository,
        model_bindings: dict | None = None,
    ) -> "BatchJobService":
        if model_bindings is None:
            model_bindings = {}

        batch_api_adapter = _get_adapter_from_model(model)
        try:
            preprocessed_inputs = await preprocessing_chain.abatch(inputs)
        except Exception as e:
            raise FailedPreProcessingError(str(e))

        batch_api_job = await batch_api_adapter.create_batch(
            preprocessed_inputs, model, model_bindings
        )

        batch_job = BatchJob(
            id=batch_api_job.id,
            provider=batch_api_job.provider,
            created_at=batch_api_job.created_at,
            postprocessing_chain=postprocessing_chain,
            metadata=batch_api_job.metadata,
        )
        await repository.save(batch_job)

        return cls(batch_api_job, batch_api_adapter, postprocessing_chain, repository)

    @classmethod
    @error_handling
    async def get(
        cls,
        job_id: str,
        repository: BatchJobRepository,
    ) -> "BatchJobService | None":
        """Resume a batch job from storage.

        Args:
            job_id: The batch job ID to resume
            repository: The repository to load from

        Returns:
            BatchJobService instance or None if job not found
        """
        batch_job = await repository.get(job_id)
        if batch_job is None:
            return None

        batch_api_job = ProviderJob(
            id=batch_job.id,
            provider=batch_job.provider,
            created_at=batch_job.created_at,
            metadata=batch_job.metadata,
        )
        batch_api_adapter = _get_adapter_from_provider(batch_job.provider)
        return cls(batch_api_job, batch_api_adapter, batch_job.postprocessing_chain, repository)

    @classmethod
    @error_handling
    async def list(
        cls,
        repository: BatchJobRepository,
        pending: bool = True,
    ) -> list["BatchJobService"]:
        """List batch jobs from storage.

        Args:
            repository: The repository to load from
            pending: If True, only return unfinished jobs. If False, return all jobs.

        Returns:
            List of BatchJobService instances
        """
        batch_jobs = await repository.list(pending=pending)
        services = []
        for batch_job in batch_jobs:
            batch_api_job = ProviderJob(
                id=batch_job.id,
                provider=batch_job.provider,
                created_at=batch_job.created_at,
                metadata=batch_job.metadata,
            )
            batch_api_adapter = _get_adapter_from_provider(batch_job.provider)
            services.append(
                cls(batch_api_job, batch_api_adapter, batch_job.postprocessing_chain, repository)
            )
        return services

    async def _postprocess(self, results: list[BatchItem]) -> list[Any]:
        """Run the postprocessing chain on batch results."""

        async def _process_example_if_successful(response: BatchItem):
            if response.success:
                try:
                    return await self.postprocessing_chain.ainvoke(response.content)
                except Exception as e:
                    return FailedPostProcessingError(str(e))
            else:
                return FailedLLMOutputError(str(response.error))

        outputs = await asyncio.gather(*[_process_example_if_successful(r) for r in results])
        return outputs

    async def _mark_as_finished(self, status_info: BatchStatusInfo):
        batch_job = await self.repository.get(self.batch_api_job.id)
        if batch_job is None:
            return
        batch_job.finished = True
        batch_job.status = status_info.status
        batch_job.total = status_info.total
        batch_job.complete = status_info.completed
        await self.repository.save(batch_job)

    @error_handling
    async def get_results(self):
        batch_status_info = await self.batch_api_adapter.get_status(self.batch_api_job)
        if batch_status_info.status == BatchStatus.COMPLETED:
            results = await self.batch_api_adapter.get_results(self.batch_api_job)
            processed_results = await self._postprocess(results)
            await self._mark_as_finished(batch_status_info)
            return ProcessedResults(
                job_id=self.batch_api_job.id,
                results=processed_results,
                status_info=batch_status_info,
            )
        # mustve failed otherwise if in FINISHED_STATUSES
        elif batch_status_info.status in FINISHED_STATUSES:
            await self._mark_as_finished(batch_status_info)
            return ProcessedResults(
                job_id=self.batch_api_job.id, results=None, status_info=batch_status_info
            )
        else:
            return ProcessedResults(
                job_id=self.batch_api_job.id, results=None, status_info=batch_status_info
            )

    @error_handling
    async def cancel(self):
        batch_status_info = await self.batch_api_adapter.cancel(self.batch_api_job)
        await self._mark_as_finished(batch_status_info)
        return True
