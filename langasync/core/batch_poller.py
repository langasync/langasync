"""BatchPoller for polling pending batch jobs and yielding results."""

import asyncio
from typing import AsyncIterator

from langasync.core.batch_api import FINISHED_STATUSES
from langasync.core.batch_service import BatchJobService, ProcessedResults
from langasync.core.batch_job_repository import BatchJobRepository


class BatchPoller:
    """Polls pending batch jobs and yields results as they complete.

    Usage:
        repository = FileSystemBatchJobRepository("./batch_jobs")
        poller = BatchPoller(repository)

        async for result in poller.wait_all():
            if result.status_info.status == BatchStatus.COMPLETED:
                print(f"Job {result.job_id}: {len(result.results)} results")
            else:
                print(f"Job {result.job_id} failed: {result.status_info.status}")
    """

    def __init__(
        self,
        repository: BatchJobRepository,
        poll_interval: float = 60.0,
    ):
        """Initialize the poller.

        Args:
            repository: Repository to load pending jobs from
            poll_interval: Seconds between status checks (default 60s)
        """
        self.repository = repository
        self.poll_interval = poll_interval

    async def _get_new_pending_services_to_watch_dict(self):
        pending_services = await BatchJobService.list(self.repository, pending=True)
        return {service.job_id: service for service in pending_services}

    async def wait_all(self, watch_for_new: bool = False) -> AsyncIterator[ProcessedResults]:
        """Poll all pending jobs and yield results as they complete.

        Args:
            watch_for_new: If True, keep running and watch for new jobs.
                          If False (default), exit when all current jobs complete.

        Yields:
            ProcessedResults for each completed job
        """
        services_to_watch = await self._get_new_pending_services_to_watch_dict()

        if watch_for_new:
            is_continue_fn = lambda _: True
        else:
            is_continue_fn = lambda _services_to_watch: len(_services_to_watch) > 0

        while is_continue_fn(services_to_watch):
            completed = []
            for job_id, service in services_to_watch.items():
                result = await service.get_results()
                if result.status_info.status in FINISHED_STATUSES:
                    completed.append((job_id, result))

            for job_id, result in completed:
                # clear up space for memory
                del services_to_watch[job_id]
                yield result

            if watch_for_new:
                # new services to watch that may have been recently submitted
                new_services_to_watch = await self._get_new_pending_services_to_watch_dict()
                services_to_watch = {**services_to_watch, **new_services_to_watch}

            # sleep before next iteration
            await asyncio.sleep(self.poll_interval)
