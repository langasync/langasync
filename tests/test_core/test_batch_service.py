"""Tests for BatchJobService and BatchJobHandle."""

import pytest
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_service import BatchJobService
from langasync.core.batch_handle import BatchJobHandle, ProcessedResults
from langasync.exceptions import (
    FailedLLMOutputError,
    FailedPostProcessingError,
    FailedPreProcessingError,
)
from langasync.core.batch_job_repository import BatchJob, FileSystemBatchJobRepository
from langasync.providers.interface import (
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchStatus,
    BatchStatusInfo,
    BatchItem,
)
from langasync.providers.none import NoModelProviderJobAdapter


class MockInProgressApiAdapter(ProviderJobAdapterInterface):
    """Mock adapter that always returns IN_PROGRESS status."""

    def __init__(self, total: int = 10, completed: int = 5):
        self.total = total
        self.completed = completed
        self.cancel_called_with: ProviderJob | None = None

    async def create_batch(self, inputs, language_model) -> ProviderJob:
        raise NotImplementedError("Not needed for these tests")

    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS,
            total=self.total,
            completed=self.completed,
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        return []

    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        return []

    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        self.cancel_called_with = batch_api_job
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=0,
            completed=0,
            failed=0,
        )


class MockFailedApiAdapter(ProviderJobAdapterInterface):
    """Mock adapter that returns FAILED status."""

    def __init__(self, status: BatchStatus = BatchStatus.FAILED, total: int = 10, failed: int = 10):
        self._status = status
        self.total = total
        self.failed = failed

    async def create_batch(self, inputs, language_model) -> ProviderJob:
        raise NotImplementedError("Not needed for these tests")

    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=self._status,
            total=self.total,
            completed=0,
            failed=self.failed,
        )

    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        return []

    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        return []

    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=self.total,
            completed=0,
            failed=self.failed,
        )


class MockAdapterWithFailedResponse(ProviderJobAdapterInterface):
    """Mock adapter that returns a mix of successful and failed responses."""

    def __init__(self, failed_indices: list[int] = None):
        self.failed_indices = failed_indices if failed_indices is not None else [1]

    async def create_batch(self, inputs, language_model) -> ProviderJob:
        raise NotImplementedError()

    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total=3,
            completed=3,
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        return []

    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        return [
            BatchItem(
                custom_id=str(i),
                success=(i not in self.failed_indices),
                content=f"result_{i}" if i not in self.failed_indices else None,
                error={"message": f"Error for item {i}"} if i in self.failed_indices else None,
            )
            for i in range(3)
        ]

    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=3,
            completed=0,
            failed=0,
        )


@pytest.fixture
def repository(test_settings) -> FileSystemBatchJobRepository:
    """Create a repository using test settings."""
    return FileSystemBatchJobRepository(test_settings)


@pytest.fixture
def batch_job_service(test_settings) -> BatchJobService:
    """Create a BatchJobService using test settings."""
    return BatchJobService(test_settings)


@pytest.fixture
def preprocessing_chain():
    """Simple preprocessing chain that passes through."""
    return RunnablePassthrough()


@pytest.fixture
def postprocessing_chain():
    """Simple postprocessing chain that converts to uppercase."""
    return RunnableLambda(lambda x: x.upper())


class TestBatchJobServiceCreate:
    """Tests for BatchJobService.create()."""

    @pytest.mark.asyncio
    async def test_create_returns_handle(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """create() returns a BatchJobHandle instance."""
        inputs = ["hello", "world"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        assert isinstance(handle, BatchJobHandle)

    @pytest.mark.asyncio
    async def test_create_assigns_job_id(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """create() assigns a job_id to the handle."""
        inputs = ["hello", "world"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        assert handle.job_id is not None
        assert handle.job_id.startswith("no-model-")

    @pytest.mark.asyncio
    async def test_create_saves_job_to_repository(
        self, batch_job_service, repository, preprocessing_chain, postprocessing_chain
    ):
        """create() persists the job to the repository."""
        inputs = ["hello", "world"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        saved_job = await repository.get(handle.job_id)
        assert saved_job is not None
        assert saved_job.id == handle.job_id
        assert saved_job.provider == "none"
        assert saved_job.finished is False

    @pytest.mark.asyncio
    async def test_create_runs_preprocessing_chain(self, batch_job_service, postprocessing_chain):
        """create() runs the preprocessing chain on inputs."""
        preprocessing = RunnableLambda(lambda x: f"preprocessed:{x}")
        inputs = ["a", "b", "c"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing,
            postprocessing_chain=postprocessing_chain,
        )

        # NoModelProviderJobAdapter stores the preprocessed inputs
        # We can verify by getting results
        result = await handle.get_results()
        assert result.results is not None
        # The postprocessing converts to uppercase
        assert "PREPROCESSED:A" in result.results
        assert "PREPROCESSED:B" in result.results
        assert "PREPROCESSED:C" in result.results

    @pytest.mark.asyncio
    async def test_create_multiple_handles_get_unique_ids(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """Each created handle gets a unique job_id."""
        handle1 = await batch_job_service.create(
            inputs=["a"],
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )
        handle2 = await batch_job_service.create(
            inputs=["b"],
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        assert handle1.job_id != handle2.job_id


class TestBatchJobServiceGet:
    """Tests for BatchJobService.get()."""

    @pytest.mark.asyncio
    async def test_get_returns_handle_for_existing_job(
        self, batch_job_service, repository, postprocessing_chain
    ):
        """get() returns a BatchJobHandle for an existing job."""
        # Manually save a job to the repository
        job = BatchJob(
            id="test-job-123",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        handle = await batch_job_service.get("test-job-123")

        assert handle is not None
        assert handle.job_id == "test-job-123"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_nonexistent_job(self, batch_job_service):
        """get() returns None for a job that doesn't exist."""
        handle = await batch_job_service.get("nonexistent-job")
        assert handle is None

    @pytest.mark.asyncio
    async def test_get_restores_postprocessing_chain(self, batch_job_service, repository):
        """get() restores the postprocessing chain from the repository."""
        chain = RunnableLambda(lambda x: x.lower())
        job = BatchJob(
            id="chain-test-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=False,
        )
        await repository.save(job)

        handle = await batch_job_service.get("chain-test-job")

        assert handle is not None
        assert isinstance(handle.postprocessing_chain, StrOutputParser)

    @pytest.mark.asyncio
    async def test_get_creates_correct_adapter(
        self, batch_job_service, repository, postprocessing_chain
    ):
        """get() creates the correct adapter for the provider."""
        job = BatchJob(
            id="adapter-test-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        handle = await batch_job_service.get("adapter-test-job")

        assert handle is not None
        assert isinstance(handle.batch_api_adapter, NoModelProviderJobAdapter)


class TestBatchJobServiceList:
    """Tests for BatchJobService.list()."""

    @pytest.mark.asyncio
    async def test_list_returns_empty_for_empty_repository(self, batch_job_service):
        """list() returns empty list when repository is empty."""
        handles = await batch_job_service.list()
        assert handles == []

    @pytest.mark.asyncio
    async def test_list_pending_returns_only_unfinished(
        self, batch_job_service, repository, postprocessing_chain
    ):
        """list(pending=True) returns only unfinished jobs."""
        pending_job = BatchJob(
            id="pending-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        finished_job = BatchJob(
            id="finished-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=True,
        )
        await repository.save(pending_job)
        await repository.save(finished_job)

        handles = await batch_job_service.list(pending=True)

        assert len(handles) == 1
        assert handles[0].job_id == "pending-job"

    @pytest.mark.asyncio
    async def test_list_all_returns_all_jobs(
        self, batch_job_service, repository, postprocessing_chain
    ):
        """list(pending=False) returns all jobs."""
        pending_job = BatchJob(
            id="pending-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        finished_job = BatchJob(
            id="finished-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=True,
        )
        await repository.save(pending_job)
        await repository.save(finished_job)

        handles = await batch_job_service.list(pending=False)

        assert len(handles) == 2
        job_ids = {s.job_id for s in handles}
        assert job_ids == {"pending-job", "finished-job"}

    @pytest.mark.asyncio
    async def test_list_default_is_pending(
        self, batch_job_service, repository, postprocessing_chain
    ):
        """list() defaults to pending=True."""
        pending_job = BatchJob(
            id="pending-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        finished_job = BatchJob(
            id="finished-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=True,
        )
        await repository.save(pending_job)
        await repository.save(finished_job)

        handles = await batch_job_service.list()

        assert len(handles) == 1
        assert handles[0].job_id == "pending-job"


class TestBatchJobHandleGetResults:
    """Tests for BatchJobHandle.get_results()."""

    @pytest.mark.asyncio
    async def test_get_results_returns_processed_results(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """get_results() returns ProcessedResults with processed data."""
        inputs = ["hello", "world"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        result = await handle.get_results()

        assert isinstance(result, ProcessedResults)
        assert result.job_id == handle.job_id
        assert result.results is not None
        # NoModelProviderJobAdapter passes through inputs as strings
        # postprocessing_chain converts to uppercase
        assert "HELLO" in result.results
        assert "WORLD" in result.results

    @pytest.mark.asyncio
    async def test_get_results_includes_status_info(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """get_results() includes status information."""
        inputs = ["a", "b", "c"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        result = await handle.get_results()

        assert result.status_info.status == BatchStatus.COMPLETED
        assert result.status_info.total == 3
        assert result.status_info.completed == 3
        assert result.status_info.failed == 0

    @pytest.mark.asyncio
    async def test_get_results_marks_job_as_finished(
        self, batch_job_service, repository, preprocessing_chain, postprocessing_chain
    ):
        """get_results() marks the job as finished in the repository when complete."""
        inputs = ["hello"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        # Before getting results
        job_before = await repository.get(handle.job_id)
        assert job_before is not None
        assert job_before.finished is False

        await handle.get_results()

        # After getting results
        job_after = await repository.get(handle.job_id)
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_get_results_runs_postprocessing_chain(
        self, batch_job_service, preprocessing_chain
    ):
        """get_results() runs the postprocessing chain on results."""
        # Custom postprocessing that reverses strings
        postprocessing = RunnableLambda(lambda x: x[::-1])
        inputs = ["hello", "world"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing,
        )

        result = await handle.get_results()

        assert result.results is not None
        assert "olleh" in result.results
        assert "dlrow" in result.results

    @pytest.mark.asyncio
    async def test_get_results_returns_none_when_not_complete(
        self, repository, postprocessing_chain
    ):
        """get_results() returns None results when job is not complete."""
        mock_adapter = MockInProgressApiAdapter(total=10, completed=5)

        job = BatchJob(
            id="in-progress-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="in-progress-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await handle.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.IN_PROGRESS
        assert result.status_info.completed == 5


class TestBatchJobHandleCancel:
    """Tests for BatchJobHandle.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_marks_job_as_finished(
        self, batch_job_service, repository, preprocessing_chain, postprocessing_chain
    ):
        """cancel() marks the job as finished in the repository."""
        inputs = ["hello"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        # Before canceling
        job_before = await repository.get(handle.job_id)
        assert job_before is not None
        assert job_before.finished is False

        await handle.cancel()

        # After canceling
        job_after = await repository.get(handle.job_id)
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_cancel_returns_true(
        self, batch_job_service, preprocessing_chain, postprocessing_chain
    ):
        """cancel() returns True."""
        inputs = ["hello"]

        handle = await batch_job_service.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
        )

        result = await handle.cancel()

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_calls_adapter_cancel(self, repository, postprocessing_chain):
        """cancel() calls the adapter's cancel method."""
        mock_adapter = MockInProgressApiAdapter()

        job = BatchJob(
            id="cancel-test-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="cancel-test-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await handle.cancel()

        assert mock_adapter.cancel_called_with == batch_api_job


class TestBatchJobHandleFailure:
    """Tests for BatchJobHandle handling of failed batch jobs."""

    @pytest.mark.asyncio
    async def test_get_results_returns_none_when_failed(self, repository, postprocessing_chain):
        """get_results() returns None results when batch has failed."""
        mock_adapter = MockFailedApiAdapter(status=BatchStatus.FAILED)

        job = BatchJob(
            id="failed-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="failed-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await handle.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_results_marks_failed_job_as_finished(self, repository, postprocessing_chain):
        """get_results() marks failed jobs as finished in the repository."""
        mock_adapter = MockFailedApiAdapter(status=BatchStatus.FAILED)

        job = BatchJob(
            id="failed-job-2",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="failed-job-2",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await handle.get_results()

        job_after = await repository.get("failed-job-2")
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_get_results_stores_failed_status(self, repository, postprocessing_chain):
        """get_results() stores the FAILED status in the repository."""
        mock_adapter = MockFailedApiAdapter(status=BatchStatus.FAILED)

        job = BatchJob(
            id="failed-job-3",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="failed-job-3",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await handle.get_results()

        job_after = await repository.get("failed-job-3")
        assert job_after is not None
        assert job_after.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_results_handles_expired_status(self, repository, postprocessing_chain):
        """get_results() handles EXPIRED status correctly."""
        mock_adapter = MockFailedApiAdapter(status=BatchStatus.EXPIRED)

        job = BatchJob(
            id="expired-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="expired-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await handle.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.EXPIRED

        job_after = await repository.get("expired-job")
        assert job_after is not None
        assert job_after.finished is True
        assert job_after.status == BatchStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_cancel_stores_cancelled_status(self, repository, postprocessing_chain):
        """cancel() stores the CANCELLED status in the repository."""
        mock_adapter = MockInProgressApiAdapter()

        job = BatchJob(
            id="cancel-status-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="cancel-status-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await handle.cancel()

        job_after = await repository.get("cancel-status-job")
        assert job_after is not None
        assert job_after.finished is True
        assert job_after.status == BatchStatus.CANCELLED


class TestPostprocessingErrors:
    """Tests for postprocessing error handling."""

    @pytest.mark.asyncio
    async def test_failed_response_returns_failed_llm_output_error(
        self, repository, postprocessing_chain
    ):
        """When response.success=False, returns FailedLLMOutputError."""
        mock_adapter = MockAdapterWithFailedResponse(failed_indices=[1])

        job = BatchJob(
            id="failed-response-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="failed-response-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await handle.get_results()

        assert result.results[0] == "RESULT_0"
        assert isinstance(result.results[1], FailedLLMOutputError)
        assert result.results[2] == "RESULT_2"

    @pytest.mark.asyncio
    async def test_postprocessing_failure_returns_failed_postprocessing_error(self, repository):
        """When postprocessing chain raises, returns FailedPostProcessingError."""
        mock_adapter = MockAdapterWithFailedResponse(failed_indices=[])

        def failing_parser(x):
            if "1" in str(x):
                raise ValueError("Parse error for item 1")
            return x.upper()

        failing_chain = RunnableLambda(failing_parser)

        job = BatchJob(
            id="failing-postprocess-job",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=failing_chain,
            finished=False,
        )
        await repository.save(job)

        batch_api_job = ProviderJob(
            id="failing-postprocess-job",
            provider="none",
            created_at=datetime.now(),
        )
        handle = BatchJobHandle(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=failing_chain,
            repository=repository,
        )

        result = await handle.get_results()

        assert result.results[0] == "RESULT_0"
        assert isinstance(result.results[1], FailedPostProcessingError)
        assert result.results[2] == "RESULT_2"


class TestPreprocessingErrors:
    """Tests for preprocessing error handling."""

    @pytest.mark.asyncio
    async def test_preprocessing_failure_raises_failed_preprocessing_error(
        self, batch_job_service, postprocessing_chain
    ):
        """When preprocessing chain raises, raises FailedPreProcessingError."""

        def failing_preprocessor(x):
            raise ValueError("Preprocessing failed")

        failing_chain = RunnableLambda(failing_preprocessor)

        with pytest.raises(FailedPreProcessingError) as exc_info:
            await batch_job_service.create(
                inputs=["a", "b", "c"],
                model=None,
                preprocessing_chain=failing_chain,
                postprocessing_chain=postprocessing_chain,
            )

        assert "Preprocessing failed" in str(exc_info.value)
