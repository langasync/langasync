from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable

from langasync.exceptions import (
    error_handling,
    FailedPreProcessingError,
)

logger = logging.getLogger(__name__)

from langasync.providers.interface import ProviderJob
from langasync.core.batch_handle import BatchJobHandle
from langasync.settings import LangasyncSettings
from langasync.core.batch_job_repository import (
    BatchJob,
    BatchJobRepository,
    batch_job_repository_factory,
)
from langasync.providers import get_adapter_from_provider, get_adapter_from_model


class BatchJobService:
    def __init__(
        self,
        settings: LangasyncSettings,
        _repository_factory: Callable[[LangasyncSettings], BatchJobRepository] = (
            batch_job_repository_factory
        ),
    ):
        self.settings = settings
        self.repository = _repository_factory(settings)

    @error_handling
    async def create(
        self,
        inputs: list[Any],
        model: BaseLanguageModel | None,
        preprocessing_chain: Runnable,
        postprocessing_chain: Runnable,
        model_bindings: dict | None = None,
    ) -> BatchJobHandle:
        if model_bindings is None:
            model_bindings = {}

        batch_api_adapter = get_adapter_from_model(model, self.settings)
        logger.debug(f"Preprocessing {len(inputs)} inputs")
        try:
            preprocessed_inputs = await preprocessing_chain.abatch(inputs)
        except Exception as e:
            raise FailedPreProcessingError(str(e))

        batch_api_job = await batch_api_adapter.create_batch(
            preprocessed_inputs, model, model_bindings  # type: ignore[arg-type]
        )
        logger.info(
            f"Batch job created: {batch_api_job.id} (provider={batch_api_job.provider.value})"
        )

        batch_job = BatchJob(
            id=batch_api_job.id,
            provider=batch_api_job.provider,
            created_at=batch_api_job.created_at,
            postprocessing_chain=postprocessing_chain,
            metadata=batch_api_job.metadata,
        )
        await self.repository.save(batch_job)

        return BatchJobHandle(
            batch_api_job, batch_api_adapter, postprocessing_chain, self.repository
        )

    @error_handling
    async def get(
        self,
        job_id: str,
    ) -> BatchJobHandle | None:
        """Resume a batch job from storage.

        Args:
            job_id: The batch job ID to resume
            repository: The repository to load from

        Returns:
            BatchJobService instance or None if job not found
        """
        batch_job = await self.repository.get(job_id)
        if batch_job is None:
            return None

        batch_api_job = ProviderJob(
            id=batch_job.id,
            provider=batch_job.provider,
            created_at=batch_job.created_at,
            metadata=batch_job.metadata,
        )
        batch_api_adapter = get_adapter_from_provider(batch_job.provider, self.settings)
        return BatchJobHandle(
            batch_api_job, batch_api_adapter, batch_job.postprocessing_chain, self.repository
        )

    @error_handling
    async def list(
        self,
        pending: bool = True,
    ) -> list[BatchJobHandle]:
        """List batch jobs from storage.

        Args:
            repository: The repository to load from
            pending: If True, only return unfinished jobs. If False, return all jobs.

        Returns:
            List of BatchJobService instances
        """
        logger.info(f"Listing batch jobs (pending={pending})")
        batch_jobs = await self.repository.list(pending=pending)
        handles = []
        for batch_job in batch_jobs:
            batch_api_job = ProviderJob(
                id=batch_job.id,
                provider=batch_job.provider,
                created_at=batch_job.created_at,
                metadata=batch_job.metadata,
            )
            batch_api_adapter = get_adapter_from_provider(batch_job.provider, self.settings)
            handles.append(
                BatchJobHandle(
                    batch_api_job,
                    batch_api_adapter,
                    batch_job.postprocessing_chain,
                    self.repository,
                )
            )
        return handles
