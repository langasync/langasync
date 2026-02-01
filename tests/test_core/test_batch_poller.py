"""Tests for BatchPoller."""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_poller import BatchPoller
from langasync.core.batch_service import BatchJobService, ProcessedResults
from langasync.core.batch_job_repository import BatchJob, FileSystemBatchJobRepository
from langasync.core.batch_api import (
    BatchStatus,
    BatchStatusInfo,
    BatchApiJob,
    BatchResponse,
)
from langasync.providers import Provider
from langasync.providers.no_provider import NoModelBatchApiAdapter


class FailingBatchApiAdapter(NoModelBatchApiAdapter):
    """Adapter that returns FAILED status for testing."""

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(status=BatchStatus.FAILED, total=1, completed=0, failed=1)

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        # Return empty results for failed job
        return []


class DelayedCompletionAdapter(NoModelBatchApiAdapter):
    """Adapter that returns IN_PROGRESS for N calls, then COMPLETED."""

    def __init__(self, calls_until_complete: int = 2):
        super().__init__()
        self.call_counts: dict[str, int] = {}
        self.calls_until_complete = calls_until_complete

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        job_id = batch_api_job.id
        self.call_counts[job_id] = self.call_counts.get(job_id, 0) + 1

        if self.call_counts[job_id] >= self.calls_until_complete:
            return BatchStatusInfo(status=BatchStatus.COMPLETED, total=1, completed=1, failed=0)
        return BatchStatusInfo(status=BatchStatus.IN_PROGRESS, total=1, completed=0, failed=0)


@pytest.fixture
def repository(tmp_path: Path) -> FileSystemBatchJobRepository:
    """Create a repository using a temporary directory."""
    return FileSystemBatchJobRepository(tmp_path)


class TestBatchPollerInit:
    """Tests for BatchPoller initialization."""

    def test_init_stores_repository(self, repository: FileSystemBatchJobRepository):
        """BatchPoller stores the repository."""
        poller = BatchPoller(repository)
        assert poller.repository is repository

    def test_init_default_poll_interval(self, repository: FileSystemBatchJobRepository):
        """BatchPoller has default poll_interval of 60 seconds."""
        poller = BatchPoller(repository)
        assert poller.poll_interval == 60.0

    def test_init_custom_poll_interval(self, repository: FileSystemBatchJobRepository):
        """BatchPoller accepts custom poll_interval."""
        poller = BatchPoller(repository, poll_interval=5.0)
        assert poller.poll_interval == 5.0


class TestBatchPollerWaitAll:
    """Tests for BatchPoller.wait_all()."""

    @pytest.mark.asyncio
    async def test_wait_all_yields_nothing_when_no_pending_jobs(
        self, repository: FileSystemBatchJobRepository
    ):
        """wait_all() completes immediately when no pending jobs."""
        poller = BatchPoller(repository, poll_interval=0.01)

        results = [r async for r in poller.wait_all()]

        assert results == []

    @pytest.mark.asyncio
    async def test_wait_all_yields_completed_job_results(
        self, repository: FileSystemBatchJobRepository
    ):
        """wait_all() yields results from completed jobs."""
        # Create a job using NoModelBatchApiAdapter (completes immediately)
        chain = RunnableLambda(lambda x: x.upper())
        service = await BatchJobService.create(
            inputs=["hello", "world"],
            model=None,
            preprocessing_chain=RunnablePassthrough(),
            postprocessing_chain=chain,
            repository=repository,
        )
        job_id = service.job_id

        poller = BatchPoller(repository, poll_interval=0.01)
        results = [r async for r in poller.wait_all()]

        assert len(results) == 1
        assert results[0].job_id == job_id
        assert results[0].results is not None
        assert "HELLO" in results[0].results
        assert "WORLD" in results[0].results

    @pytest.mark.asyncio
    async def test_wait_all_yields_multiple_job_results(
        self, repository: FileSystemBatchJobRepository
    ):
        """wait_all() yields results from multiple completed jobs."""
        chain = RunnablePassthrough()

        service1 = await BatchJobService.create(
            inputs=["a"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )
        service2 = await BatchJobService.create(
            inputs=["b"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        poller = BatchPoller(repository, poll_interval=0.01)
        results = [r async for r in poller.wait_all()]

        assert len(results) == 2
        job_ids = {r.job_id for r in results}
        assert service1.job_id in job_ids
        assert service2.job_id in job_ids

    @pytest.mark.asyncio
    async def test_wait_all_marks_jobs_as_finished(self, repository: FileSystemBatchJobRepository):
        """wait_all() marks jobs as finished after yielding results."""
        chain = RunnablePassthrough()
        service = await BatchJobService.create(
            inputs=["test"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        # Before polling
        job_before = await repository.get(service.job_id)
        assert job_before is not None
        assert job_before.finished is False

        poller = BatchPoller(repository, poll_interval=0.01)
        results = [r async for r in poller.wait_all()]

        # After polling
        job_after = await repository.get(service.job_id)
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_wait_all_ignores_finished_jobs(self, repository: FileSystemBatchJobRepository):
        """wait_all() only processes pending jobs, not finished ones."""
        # Create a finished job directly
        finished_job = BatchJob(
            id="finished-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=RunnablePassthrough(),
            finished=True,
        )
        await repository.save(finished_job)

        # Create a pending job
        chain = RunnablePassthrough()
        pending_service = await BatchJobService.create(
            inputs=["pending"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        poller = BatchPoller(repository, poll_interval=0.01)
        results = [r async for r in poller.wait_all()]

        # Only the pending job should be yielded
        assert len(results) == 1
        assert results[0].job_id == pending_service.job_id

    @pytest.mark.asyncio
    async def test_wait_all_returns_processed_results(
        self, repository: FileSystemBatchJobRepository
    ):
        """wait_all() returns ProcessedResults objects."""
        chain = RunnablePassthrough()
        await BatchJobService.create(
            inputs=["test"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        poller = BatchPoller(repository, poll_interval=0.01)
        results = [r async for r in poller.wait_all()]

        assert len(results) == 1
        assert isinstance(results[0], ProcessedResults)
        assert results[0].status_info.status == BatchStatus.COMPLETED


class TestBatchPollerContinuousMode:
    """Tests for BatchPoller.wait_all(watch_for_new=True)."""

    @pytest.mark.asyncio
    async def test_continuous_mode_picks_up_new_jobs(
        self, repository: FileSystemBatchJobRepository
    ):
        """Poller picks up jobs added after receiving first result."""
        chain = RunnablePassthrough()
        poller = BatchPoller(repository, poll_interval=0.01)
        results = []

        # Create first job before starting
        await BatchJobService.create(
            inputs=["first"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        async for result in poller.wait_all(watch_for_new=True):
            results.append(result)
            if len(results) < 3:
                # Create another job after we got prior result
                await BatchJobService.create(
                    inputs=["second"],
                    model=None,
                    preprocessing_chain=chain,
                    postprocessing_chain=chain,
                    repository=repository,
                )
            else:
                break

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_continuous_mode_keeps_running_when_empty(
        self, repository: FileSystemBatchJobRepository
    ):
        """Poller keeps running even when no jobs exist initially."""
        chain = RunnablePassthrough()
        poller = BatchPoller(repository, poll_interval=0.01)

        # This test needs asyncio.gather since we need to add job while poller runs
        async def collect_results():
            async for result in poller.wait_all(watch_for_new=True):
                results.append(result)
                break

        async def add_job_later():
            await asyncio.sleep(0.5)  # Small delay to ensure poller started
            await BatchJobService.create(
                inputs=["delayed"],
                model=None,
                preprocessing_chain=chain,
                postprocessing_chain=chain,
                repository=repository,
            )

        results = []
        await asyncio.gather(
            collect_results(),
            add_job_later(),
        )

        assert len(results) == 1
        assert "delayed" in results[0].results


class TestBatchPollerFailedJobs:
    """Tests for BatchPoller handling of failed jobs."""

    @pytest.mark.asyncio
    async def test_poller_yields_failed_jobs(self, repository: FileSystemBatchJobRepository):
        """Poller yields jobs with FAILED status instead of polling forever."""
        chain = RunnablePassthrough()
        await BatchJobService.create(
            inputs=["test"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        # Patch registry to use failing adapter when resuming job
        with patch.dict(
            "langasync.providers.ADAPTER_REGISTRY",
            {Provider.NONE: FailingBatchApiAdapter},
        ):
            poller = BatchPoller(repository, poll_interval=0.01)
            results = [r async for r in poller.wait_all()]

        assert len(results) == 1
        assert results[0].status_info.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_failed_job_marked_as_finished(self, repository: FileSystemBatchJobRepository):
        """Failed jobs are marked as finished in the repository."""
        chain = RunnablePassthrough()
        service = await BatchJobService.create(
            inputs=["test"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        with patch.dict(
            "langasync.providers.ADAPTER_REGISTRY",
            {Provider.NONE: FailingBatchApiAdapter},
        ):
            poller = BatchPoller(repository, poll_interval=0.01)
            _ = [r async for r in poller.wait_all()]

        # Verify job is marked finished even though it failed
        job = await repository.get(service.job_id)
        assert job is not None
        assert job.finished is True


class TestBatchPollerDelayedCompletion:
    """Tests for BatchPoller handling jobs that take multiple polls to complete."""

    @pytest.mark.asyncio
    async def test_poller_waits_for_in_progress_jobs(
        self, repository: FileSystemBatchJobRepository
    ):
        """Poller keeps polling IN_PROGRESS jobs until they complete."""
        chain = RunnablePassthrough()
        await BatchJobService.create(
            inputs=["test"],
            model=None,
            preprocessing_chain=chain,
            postprocessing_chain=chain,
            repository=repository,
        )

        # Create adapter instance with shared state we can inspect
        adapter = DelayedCompletionAdapter(calls_until_complete=3)

        with patch(
            "langasync.core.batch_service._get_adapter_from_provider",
            return_value=adapter,
        ):
            poller = BatchPoller(repository, poll_interval=0.01)
            results = [r async for r in poller.wait_all()]

        assert len(results) == 1
        assert results[0].status_info.status == BatchStatus.COMPLETED
        # Verify it was polled multiple times before completing
        assert len(adapter.call_counts) == 1
        job_id = list(adapter.call_counts.keys())[0]
        assert adapter.call_counts[job_id] >= 3
