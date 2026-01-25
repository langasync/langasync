import asyncio
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable

from langasync.core.exceptions import UnsupportedProviderError
from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchApiJob,
    FINISHED_STATUSES,
    BatchResponse,
)
from langasync.core.batch_job_repository import BatchJob, BatchJobRepository
from langasync.core.get_parts_from_chain import get_parts_from_chain
from langasync.providers import ADAPTER_REGISTRY, Provider


def _get_adapter_from_provider(provider: Provider) -> BatchApiAdapterInterface:
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


def _get_adapter_from_model(model: BaseLanguageModel | None) -> BatchApiAdapterInterface:
    """Get the appropriate batch API adapter for a model."""
    provider = _get_provider_from_model(model)
    return _get_adapter_from_provider(provider)


class BatchJobService:
    def __init__(
        self,
        batch_api_job: BatchApiJob,
        batch_api_adapter: BatchApiAdapterInterface,
        postprocessing_chain: Runnable,
        repository: BatchJobRepository,
    ):
        self.batch_api_job = batch_api_job
        self.batch_api_adapter = batch_api_adapter
        self.postprocessing_chain = postprocessing_chain
        self.repository = repository

    async def _postprocess(self, results: list[BatchResponse]) -> list[Any]:
        """Run the postprocessing chain on batch results."""
        contents = [r.content for r in results]
        return await self.postprocessing_chain.abatch(contents)

    @classmethod
    async def create(
        cls,
        inputs: list[Any],
        model: BaseLanguageModel | None,
        preprocessing_chain: Runnable,
        postprocessing_chain: Runnable,
        repository: BatchJobRepository,
    ) -> "BatchJobService":
        batch_api_adapter = _get_adapter_from_model(model)
        preprocessed_inputs = await preprocessing_chain.abatch(inputs)
        batch_api_job = await batch_api_adapter.create_batch(preprocessed_inputs, model)

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

        batch_api_job = BatchApiJob(
            id=batch_job.id,
            provider=batch_job.provider,
            created_at=batch_job.created_at,
            metadata=batch_job.metadata,
        )
        batch_api_adapter = _get_adapter_from_provider(batch_job.provider)
        return cls(batch_api_job, batch_api_adapter, batch_job.postprocessing_chain, repository)

    async def get_results(self):
        batch_status = await self.batch_api_adapter.get_status(self.batch_api_job)
        if batch_status.status in FINISHED_STATUSES:
            results = await self.batch_api_adapter.get_results(self.batch_api_job)
            return await self._postprocess(results)
        else:
            return None

    async def get_status(self):
        return await self.batch_api_adapter.get_status(self.batch_api_job)

    async def cancel(self):
        await self.batch_api_adapter.cancel(self.batch_api_job)
        return True


class BatchChainWrapper:
    def __init__(self, chain, repository: BatchJobRepository):
        parts = get_parts_from_chain(chain)

        self.model = parts.model
        self.preprocessing_chain = parts.preprocessing
        self.postprocessing_chain = parts.postprocessing
        self.repository = repository

    async def submit(self, inputs) -> BatchJobService:
        return await BatchJobService.create(
            inputs,
            self.model,
            self.preprocessing_chain,
            self.postprocessing_chain,
            self.repository,
        )


def batch_chain(chain, repository: BatchJobRepository) -> BatchChainWrapper:
    return BatchChainWrapper(chain, repository)
