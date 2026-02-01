"""Tests for FileSystemBatchJobRepository."""

import base64
import json
import pytest
from datetime import datetime
from pathlib import Path

import cloudpickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_job_repository import BatchJob, FileSystemBatchJobRepository


@pytest.fixture
def repository(tmp_path: Path) -> FileSystemBatchJobRepository:
    """Create a repository using a temporary directory."""
    return FileSystemBatchJobRepository(tmp_path)


@pytest.fixture
def sample_job() -> BatchJob:
    """Create a sample batch job for testing."""
    return BatchJob(
        id="job-123",
        provider="openai",
        created_at=datetime(2024, 1, 15, 10, 30, 0),
        postprocessing_chain=StrOutputParser(),
        finished=False,
        metadata={"model": "gpt-4", "batch_size": 100},
    )


class TestFileSystemBatchJobRepositoryInit:
    """Tests for repository initialization."""

    def test_creates_storage_directory(self, tmp_path: Path):
        """Repository creates the storage directory if it doesn't exist."""
        storage_dir = tmp_path / "nested" / "batch_jobs"
        assert not storage_dir.exists()

        FileSystemBatchJobRepository(storage_dir)

        assert storage_dir.exists()
        assert storage_dir.is_dir()

    def test_accepts_existing_directory(self, tmp_path: Path):
        """Repository works with an existing directory."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        repo = FileSystemBatchJobRepository(tmp_path)
        assert repo.storage_dir == tmp_path

    def test_accepts_string_path(self, tmp_path: Path):
        """Repository accepts string path and converts to Path."""
        repo = FileSystemBatchJobRepository(str(tmp_path))
        assert isinstance(repo.storage_dir, Path)


class TestSave:
    """Tests for the save method."""

    @pytest.mark.asyncio
    async def test_save_creates_json_file(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Saving a job creates a JSON file with the job ID."""
        await repository.save(sample_job)

        job_file = repository.storage_dir / "job-123.json"
        assert job_file.exists()

    @pytest.mark.asyncio
    async def test_save_persists_all_fields(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Saved JSON contains all job fields."""
        await repository.save(sample_job)

        job_file = repository.storage_dir / "job-123.json"
        data = json.loads(job_file.read_text())

        assert data["id"] == "job-123"
        assert data["provider"] == "openai"
        assert data["created_at"] == "2024-01-15T10:30:00"
        assert data["finished"] is False
        assert data["metadata"] == {"model": "gpt-4", "batch_size": 100}
        assert "postprocessing_chain" in data

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Saving a job with the same ID overwrites the existing file."""
        await repository.save(sample_job)

        sample_job.finished = True
        sample_job.metadata["updated"] = True
        await repository.save(sample_job)

        retrieved = await repository.get("job-123")
        assert retrieved is not None
        assert retrieved.finished is True
        assert retrieved.metadata["updated"] is True


class TestGet:
    """Tests for the get method."""

    @pytest.mark.asyncio
    async def test_get_returns_saved_job(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Getting a saved job returns the correct data."""
        await repository.save(sample_job)

        retrieved = await repository.get("job-123")

        assert retrieved is not None
        assert retrieved.id == "job-123"
        assert retrieved.provider == "openai"
        assert retrieved.created_at == datetime(2024, 1, 15, 10, 30, 0)
        assert retrieved.finished is False
        assert retrieved.metadata == {"model": "gpt-4", "batch_size": 100}

    @pytest.mark.asyncio
    async def test_get_returns_none_for_nonexistent(self, repository: FileSystemBatchJobRepository):
        """Getting a non-existent job returns None."""
        result = await repository.get("nonexistent-job")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_deserializes_postprocessing_chain(
        self, repository: FileSystemBatchJobRepository
    ):
        """The postprocessing chain is properly deserialized."""
        job = BatchJob(
            id="job-with-chain",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
        )
        await repository.save(job)

        retrieved = await repository.get("job-with-chain")
        assert retrieved is not None
        assert isinstance(retrieved.postprocessing_chain, StrOutputParser)

    @pytest.mark.asyncio
    async def test_get_handles_missing_optional_fields(
        self, repository: FileSystemBatchJobRepository, tmp_path: Path
    ):
        """Jobs saved without optional fields get default values."""
        # Serialize the chain using cloudpickle
        chain = StrOutputParser()
        chain_bytes = cloudpickle.dumps(chain)
        chain_b64 = base64.b64encode(chain_bytes).decode("utf-8")

        # Manually create a minimal JSON file without optional fields
        job_file = tmp_path / "minimal-job.json"
        job_file.write_text(
            json.dumps(
                {
                    "id": "minimal-job",
                    "provider": "openai",
                    "created_at": "2024-01-15T10:30:00",
                    "postprocessing_chain": chain_b64,
                }
            )
        )

        retrieved = await repository.get("minimal-job")

        assert retrieved is not None
        assert retrieved.finished is False
        assert retrieved.metadata == {}


class TestList:
    """Tests for the list method."""

    @pytest.mark.asyncio
    async def test_list_empty_repository(self, repository: FileSystemBatchJobRepository):
        """Listing an empty repository returns empty list."""
        result = await repository.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_pending_returns_unfinished_jobs(
        self, repository: FileSystemBatchJobRepository
    ):
        """list(pending=True) returns only unfinished jobs."""
        pending_job = BatchJob(
            id="pending",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=False,
        )
        finished_job = BatchJob(
            id="finished",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=True,
        )

        await repository.save(pending_job)
        await repository.save(finished_job)

        result = await repository.list(pending=True)

        assert len(result) == 1
        assert result[0].id == "pending"

    @pytest.mark.asyncio
    async def test_list_all_returns_all_jobs(self, repository: FileSystemBatchJobRepository):
        """list(pending=False) returns all jobs including finished."""
        pending_job = BatchJob(
            id="pending",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=False,
        )
        finished_job = BatchJob(
            id="finished",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=True,
        )

        await repository.save(pending_job)
        await repository.save(finished_job)

        result = await repository.list(pending=False)

        assert len(result) == 2
        ids = {job.id for job in result}
        assert ids == {"pending", "finished"}

    @pytest.mark.asyncio
    async def test_list_default_is_pending(self, repository: FileSystemBatchJobRepository):
        """list() defaults to pending=True."""
        pending_job = BatchJob(
            id="pending",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=False,
        )
        finished_job = BatchJob(
            id="finished",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            finished=True,
        )

        await repository.save(pending_job)
        await repository.save(finished_job)

        result = await repository.list()

        assert len(result) == 1
        assert result[0].id == "pending"

    @pytest.mark.asyncio
    async def test_list_multiple_pending_jobs(self, repository: FileSystemBatchJobRepository):
        """list() returns multiple pending jobs."""
        for i in range(5):
            job = BatchJob(
                id=f"job-{i}",
                provider="openai",
                created_at=datetime.now(),
                postprocessing_chain=StrOutputParser(),
                finished=False,
            )
            await repository.save(job)

        result = await repository.list(pending=True)
        assert len(result) == 5


class TestDelete:
    """Tests for the delete method."""

    @pytest.mark.asyncio
    async def test_delete_removes_job_file(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Deleting a job removes its file."""
        await repository.save(sample_job)
        job_file = repository.storage_dir / "job-123.json"
        assert job_file.exists()

        await repository.delete("job-123")

        assert not job_file.exists()

    @pytest.mark.asyncio
    async def test_delete_job_no_longer_retrievable(
        self, repository: FileSystemBatchJobRepository, sample_job: BatchJob
    ):
        """Deleted job cannot be retrieved."""
        await repository.save(sample_job)
        await repository.delete("job-123")

        result = await repository.get("job-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_job_is_noop(self, repository: FileSystemBatchJobRepository):
        """Deleting a non-existent job doesn't raise an error."""
        await repository.delete("nonexistent-job")

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, repository: FileSystemBatchJobRepository):
        """Deleted jobs don't appear in list results."""
        job1 = BatchJob(
            id="job-1",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
        )
        job2 = BatchJob(
            id="job-2",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
        )

        await repository.save(job1)
        await repository.save(job2)
        await repository.delete("job-1")

        result = await repository.list(pending=False)
        assert len(result) == 1
        assert result[0].id == "job-2"


class TestRoundTrip:
    """Tests for serialization round-trip."""

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_datetime_precision(
        self, repository: FileSystemBatchJobRepository
    ):
        """Datetime is preserved through save/get cycle."""
        original_time = datetime(2024, 6, 15, 14, 30, 45)
        job = BatchJob(
            id="datetime-test",
            provider="openai",
            created_at=original_time,
            postprocessing_chain=StrOutputParser(),
        )

        await repository.save(job)
        retrieved = await repository.get("datetime-test")

        assert retrieved is not None
        assert retrieved.created_at == original_time

    @pytest.mark.asyncio
    async def test_roundtrip_complex_metadata(self, repository: FileSystemBatchJobRepository):
        """Complex metadata is preserved."""
        job = BatchJob(
            id="metadata-test",
            provider="anthropic",
            created_at=datetime.now(),
            postprocessing_chain=StrOutputParser(),
            metadata={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "number": 42,
                "boolean": True,
                "null": None,
            },
        )

        await repository.save(job)
        retrieved = await repository.get("metadata-test")

        assert retrieved is not None
        assert retrieved.metadata == {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
            "boolean": True,
            "null": None,
        }

    @pytest.mark.asyncio
    async def test_roundtrip_passthrough_chain(self, repository: FileSystemBatchJobRepository):
        """RunnablePassthrough chain is preserved."""
        job = BatchJob(
            id="passthrough-test",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=RunnablePassthrough(),
        )

        await repository.save(job)
        retrieved = await repository.get("passthrough-test")

        assert retrieved is not None
        assert isinstance(retrieved.postprocessing_chain, RunnablePassthrough)

    @pytest.mark.asyncio
    async def test_roundtrip_chained_runnables(self, repository: FileSystemBatchJobRepository):
        """Chained runnables are preserved."""
        chain = RunnablePassthrough() | StrOutputParser()
        job = BatchJob(
            id="chain-test",
            provider="openai",
            created_at=datetime.now(),
            postprocessing_chain=chain,
        )

        await repository.save(job)
        retrieved = await repository.get("chain-test")

        assert retrieved is not None
        # Verify the chain works
        result = retrieved.postprocessing_chain.invoke("test input")
        assert result == "test input"
