from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

from langasync.exceptions import (
    error_handling,
    LangAsyncError,
    FailedLLMOutputError,
    FailedPostProcessingError,
)

from langasync.providers.interface import (
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchStatusInfo,
    FINISHED_STATUSES,
    BatchStatus,
    BatchItem,
)
from langasync.core.batch_job_repository import (
    BatchJobRepository,
)


# NOTE: outputting dataclass because we want to allow error types
# and dont plan on serialising this
@dataclass
class ProcessedResults:
    """Result from get_results(), includes status and processed outputs."""

    job_id: str
    results: list[Any | LangAsyncError] | None
    status_info: BatchStatusInfo


class BatchJobHandle:
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

    async def _postprocess(self, results: list[BatchItem]) -> list[Any]:
        """Run the postprocessing chain on batch results."""

        # Returns errors instead of raising to support partial batch completions
        async def _process_example_if_successful(response: BatchItem) -> Any:
            if response.success:
                try:
                    return await self.postprocessing_chain.ainvoke(response.content)
                except Exception as e:
                    logger.error(
                        f"Job {self.batch_api_job.id}: postprocessing failed for item: {e}"
                    )
                    return FailedPostProcessingError(str(e))
            else:
                return FailedLLMOutputError(str(response.error))

        outputs = await asyncio.gather(*[_process_example_if_successful(r) for r in results])
        return outputs

    async def _mark_as_finished(self, status_info: BatchStatusInfo) -> None:
        batch_job = await self.repository.get(self.batch_api_job.id)
        if batch_job is None:
            logger.warning(f"Job {self.batch_api_job.id}: not found in repository")
            return
        batch_job.finished = True
        batch_job.status = status_info.status
        batch_job.total = status_info.total
        batch_job.complete = status_info.completed
        await self.repository.save(batch_job)

    @error_handling
    async def get_results(self) -> ProcessedResults:
        batch_status_info = await self.batch_api_adapter.get_status(self.batch_api_job)
        logger.info(f"Job {self.batch_api_job.id}: status={batch_status_info.status.value}")
        if batch_status_info.status == BatchStatus.COMPLETED:
            results = await self.batch_api_adapter.get_results(self.batch_api_job)
            processed_results = await self._postprocess(results)
            await self._mark_as_finished(batch_status_info)
            logger.info(
                f"Job {self.batch_api_job.id}: completed with {len(processed_results)} results"
            )
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
    async def cancel(self) -> bool:
        batch_status_info = await self.batch_api_adapter.cancel(self.batch_api_job)
        await self._mark_as_finished(batch_status_info)
        logger.info(f"Job {self.batch_api_job.id}: cancelled")
        return True
