"""Tests for BatchJobService."""

import pytest
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_service import BatchJobService, ProcessedResults
from langasync.core.exceptions import FailedLLMOutputError, FailedPostProcessingError, FailedPreProcessingError
from langasync.core.batch_job_repository import BatchJob, FileSystemBatchJobRepository
from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchApiJob,
    BatchStatus,
    BatchStatusInfo,
    BatchResponse,
)
from langasync.providers.none import NoModelBatchApiAdapter


class MockInProgressApiAdapter(BatchApiAdapterInterface):
    """Mock adapter that always returns IN_PROGRESS status."""

    def __init__(self, total: int = 10, completed: int = 5):
        self.total = total
        self.completed = completed
        self.cancel_called_with: BatchApiJob | None = None

    async def create_batch(self, inputs, language_model) -> BatchApiJob:
        raise NotImplementedError("Not needed for these tests")

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.IN_PROGRESS,
            total=self.total,
            completed=self.completed,
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        return []

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        return []

    async def cancel(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        self.cancel_called_with = batch_api_job
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=0,
            completed=0,
            failed=0,
        )


class MockFailedApiAdapter(BatchApiAdapterInterface):
    """Mock adapter that returns FAILED status."""

    def __init__(self, status: BatchStatus = BatchStatus.FAILED, total: int = 10, failed: int = 10):
        self._status = status
        self.total = total
        self.failed = failed

    async def create_batch(self, inputs, language_model) -> BatchApiJob:
        raise NotImplementedError("Not needed for these tests")

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=self._status,
            total=self.total,
            completed=0,
            failed=self.failed,
        )

    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        return []

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        return []

    async def cancel(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=self.total,
            completed=0,
            failed=self.failed,
        )


class MockAdapterWithFailedResponse(BatchApiAdapterInterface):
    """Mock adapter that returns a mix of successful and failed responses."""

    def __init__(self, failed_indices: list[int] = None):
        self.failed_indices = failed_indices if failed_indices is not None else [1]

    async def create_batch(self, inputs, language_model) -> BatchApiJob:
        raise NotImplementedError()

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total=3,
            completed=3,
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        return []

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        return [
            BatchResponse(
                custom_id=str(i),
                success=(i not in self.failed_indices),
                content=f"result_{i}" if i not in self.failed_indices else None,
                error={"message": f"Error for item {i}"} if i in self.failed_indices else None,
            )
            for i in range(3)
        ]

    async def cancel(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=3,
            completed=0,
            failed=0,
        )


@pytest.fixture
def repository(tmp_path: Path) -> FileSystemBatchJobRepository:
    """Create a repository using a temporary directory."""
    return FileSystemBatchJobRepository(tmp_path)


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
    async def test_create_returns_service(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """create() returns a BatchJobService instance."""
        inputs = ["hello", "world"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        assert isinstance(service, BatchJobService)

    @pytest.mark.asyncio
    async def test_create_assigns_job_id(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """create() assigns a job_id to the service."""
        inputs = ["hello", "world"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        assert service.job_id is not None
        assert service.job_id.startswith("no-model-")

    @pytest.mark.asyncio
    async def test_create_saves_job_to_repository(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """create() persists the job to the repository."""
        inputs = ["hello", "world"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        saved_job = await repository.get(service.job_id)
        assert saved_job is not None
        assert saved_job.id == service.job_id
        assert saved_job.provider == "none"
        assert saved_job.finished is False

    @pytest.mark.asyncio
    async def test_create_runs_preprocessing_chain(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
        """create() runs the preprocessing chain on inputs."""
        preprocessing = RunnableLambda(lambda x: f"preprocessed:{x}")
        inputs = ["a", "b", "c"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        # NoModelBatchApiAdapter stores the preprocessed inputs
        # We can verify by getting results
        result = await service.get_results()
        assert result.results is not None
        # The postprocessing converts to uppercase
        assert "PREPROCESSED:A" in result.results
        assert "PREPROCESSED:B" in result.results
        assert "PREPROCESSED:C" in result.results

    @pytest.mark.asyncio
    async def test_create_multiple_services_get_unique_ids(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """Each created service gets a unique job_id."""
        service1 = await BatchJobService.create(
            inputs=["a"],
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )
        service2 = await BatchJobService.create(
            inputs=["b"],
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        assert service1.job_id != service2.job_id


class TestBatchJobServiceGet:
    """Tests for BatchJobService.get()."""

    @pytest.mark.asyncio
    async def test_get_returns_service_for_existing_job(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
        """get() returns a BatchJobService for an existing job."""
        # Manually save a job to the repository
        job = BatchJob(
            id="test-job-123",
            provider="none",
            created_at=datetime.now(),
            postprocessing_chain=postprocessing_chain,
            finished=False,
        )
        await repository.save(job)

        service = await BatchJobService.get("test-job-123", repository)

        assert service is not None
        assert service.job_id == "test-job-123"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_nonexistent_job(
        self, repository: FileSystemBatchJobRepository
    ):
        """get() returns None for a job that doesn't exist."""
        service = await BatchJobService.get("nonexistent-job", repository)
        assert service is None

    @pytest.mark.asyncio
    async def test_get_restores_postprocessing_chain(
        self, repository: FileSystemBatchJobRepository
    ):
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

        service = await BatchJobService.get("chain-test-job", repository)

        assert service is not None
        assert isinstance(service.postprocessing_chain, StrOutputParser)

    @pytest.mark.asyncio
    async def test_get_creates_correct_adapter(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        service = await BatchJobService.get("adapter-test-job", repository)

        assert service is not None
        assert isinstance(service.batch_api_adapter, NoModelBatchApiAdapter)


class TestBatchJobServiceList:
    """Tests for BatchJobService.list()."""

    @pytest.mark.asyncio
    async def test_list_returns_empty_for_empty_repository(
        self, repository: FileSystemBatchJobRepository
    ):
        """list() returns empty list when repository is empty."""
        services = await BatchJobService.list(repository)
        assert services == []

    @pytest.mark.asyncio
    async def test_list_pending_returns_only_unfinished(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        services = await BatchJobService.list(repository, pending=True)

        assert len(services) == 1
        assert services[0].job_id == "pending-job"

    @pytest.mark.asyncio
    async def test_list_all_returns_all_jobs(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        services = await BatchJobService.list(repository, pending=False)

        assert len(services) == 2
        job_ids = {s.job_id for s in services}
        assert job_ids == {"pending-job", "finished-job"}

    @pytest.mark.asyncio
    async def test_list_default_is_pending(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        services = await BatchJobService.list(repository)

        assert len(services) == 1
        assert services[0].job_id == "pending-job"


class TestBatchJobServiceGetResults:
    """Tests for BatchJobService.get_results()."""

    @pytest.mark.asyncio
    async def test_get_results_returns_processed_results(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """get_results() returns ProcessedResults with processed data."""
        inputs = ["hello", "world"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert isinstance(result, ProcessedResults)
        assert result.job_id == service.job_id
        assert result.results is not None
        # NoModelBatchApiAdapter passes through inputs as strings
        # postprocessing_chain converts to uppercase
        assert "HELLO" in result.results
        assert "WORLD" in result.results

    @pytest.mark.asyncio
    async def test_get_results_includes_status_info(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """get_results() includes status information."""
        inputs = ["a", "b", "c"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.status_info.status == BatchStatus.COMPLETED
        assert result.status_info.total == 3
        assert result.status_info.completed == 3
        assert result.status_info.failed == 0

    @pytest.mark.asyncio
    async def test_get_results_marks_job_as_finished(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """get_results() marks the job as finished in the repository when complete."""
        inputs = ["hello"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        # Before getting results
        job_before = await repository.get(service.job_id)
        assert job_before is not None
        assert job_before.finished is False

        await service.get_results()

        # After getting results
        job_after = await repository.get(service.job_id)
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_get_results_runs_postprocessing_chain(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain
    ):
        """get_results() runs the postprocessing chain on results."""
        # Custom postprocessing that reverses strings
        postprocessing = RunnableLambda(lambda x: x[::-1])
        inputs = ["hello", "world"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results is not None
        assert "olleh" in result.results
        assert "dlrow" in result.results

    @pytest.mark.asyncio
    async def test_get_results_returns_none_when_not_complete(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        batch_api_job = BatchApiJob(
            id="in-progress-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.IN_PROGRESS
        assert result.status_info.completed == 5


class TestBatchJobServiceCancel:
    """Tests for BatchJobService.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_marks_job_as_finished(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """cancel() marks the job as finished in the repository."""
        inputs = ["hello"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        # Before canceling
        job_before = await repository.get(service.job_id)
        assert job_before is not None
        assert job_before.finished is False

        await service.cancel()

        # After canceling
        job_after = await repository.get(service.job_id)
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_cancel_returns_true(
        self, repository: FileSystemBatchJobRepository, preprocessing_chain, postprocessing_chain
    ):
        """cancel() returns True."""
        inputs = ["hello"]

        service = await BatchJobService.create(
            inputs=inputs,
            model=None,
            preprocessing_chain=preprocessing_chain,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.cancel()

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_calls_adapter_cancel(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="cancel-test-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await service.cancel()

        assert mock_adapter.cancel_called_with == batch_api_job


class TestBatchJobServiceFailure:
    """Tests for BatchJobService handling of failed batch jobs."""

    @pytest.mark.asyncio
    async def test_get_results_returns_none_when_failed(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="failed-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_results_marks_failed_job_as_finished(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="failed-job-2",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await service.get_results()

        job_after = await repository.get("failed-job-2")
        assert job_after is not None
        assert job_after.finished is True

    @pytest.mark.asyncio
    async def test_get_results_stores_failed_status(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="failed-job-3",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await service.get_results()

        job_after = await repository.get("failed-job-3")
        assert job_after is not None
        assert job_after.status == BatchStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_results_handles_expired_status(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="expired-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results is None
        assert result.status_info.status == BatchStatus.EXPIRED

        job_after = await repository.get("expired-job")
        assert job_after is not None
        assert job_after.finished is True
        assert job_after.status == BatchStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_cancel_stores_cancelled_status(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
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

        batch_api_job = BatchApiJob(
            id="cancel-status-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        await service.cancel()

        job_after = await repository.get("cancel-status-job")
        assert job_after is not None
        assert job_after.finished is True
        assert job_after.status == BatchStatus.CANCELLED


class TestPostprocessingErrors:
    """Tests for postprocessing error handling."""

    @pytest.mark.asyncio
    async def test_failed_response_returns_failed_llm_output_error(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
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

        batch_api_job = BatchApiJob(
            id="failed-response-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=postprocessing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results[0] == "RESULT_0"
        assert isinstance(result.results[1], FailedLLMOutputError)
        assert result.results[2] == "RESULT_2"

    @pytest.mark.asyncio
    async def test_postprocessing_failure_returns_failed_postprocessing_error(
        self, repository: FileSystemBatchJobRepository
    ):
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

        batch_api_job = BatchApiJob(
            id="failing-postprocess-job",
            provider="none",
            created_at=datetime.now(),
        )
        service = BatchJobService(
            batch_api_job=batch_api_job,
            batch_api_adapter=mock_adapter,
            postprocessing_chain=failing_chain,
            repository=repository,
        )

        result = await service.get_results()

        assert result.results[0] == "RESULT_0"
        assert isinstance(result.results[1], FailedPostProcessingError)
        assert result.results[2] == "RESULT_2"


class TestPreprocessingErrors:
    """Tests for preprocessing error handling."""

    @pytest.mark.asyncio
    async def test_preprocessing_failure_raises_failed_preprocessing_error(
        self, repository: FileSystemBatchJobRepository, postprocessing_chain
    ):
        """When preprocessing chain raises, raises FailedPreProcessingError."""
        def failing_preprocessor(x):
            raise ValueError("Preprocessing failed")

        failing_chain = RunnableLambda(failing_preprocessor)

        with pytest.raises(FailedPreProcessingError) as exc_info:
            await BatchJobService.create(
                inputs=["a", "b", "c"],
                model=None,
                preprocessing_chain=failing_chain,
                postprocessing_chain=postprocessing_chain,
                repository=repository,
            )

        assert "Preprocessing failed" in str(exc_info.value)
