import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cloudpickle
from langchain_core.runnables import Runnable


@dataclass
class BatchJob:
    """Persistable batch job with postprocessing chain."""

    id: str
    provider: str
    created_at: datetime
    postprocessing_chain: Runnable
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchJobRepository(ABC):
    """Abstract interface for persisting batch jobs."""

    @abstractmethod
    async def save(self, batch_job: BatchJob) -> None:
        pass

    @abstractmethod
    async def get(self, job_id: str) -> BatchJob | None:
        """Retrieve a batch job by ID.

        Args:
            job_id: The batch job ID

        Returns:
            BatchJob or None if not found
        """
        pass

    @abstractmethod
    async def list(self, pending: bool = True) -> list[BatchJob]:
        """List batch jobs.

        Args:
            pending: If True, only return unfinished jobs. If False, return all jobs.

        Returns:
            List of BatchJob instances
        """
        pass

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Remove a batch job from storage.

        Args:
            job_id: The batch job ID to delete
        """
        pass


class FileSystemBatchJobRepository(BatchJobRepository):
    """File-based implementation of BatchJobRepository.

    Stores each job as a JSON file in the specified directory.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.storage_dir / f"{job_id}.json"

    async def save(self, batch_job: BatchJob) -> None:
        # Use cloudpickle to serialize the chain (supports RunnableLambda with lambdas)
        chain_bytes = cloudpickle.dumps(batch_job.postprocessing_chain)
        chain_b64 = base64.b64encode(chain_bytes).decode("utf-8")

        data = {
            "id": batch_job.id,
            "provider": batch_job.provider,
            "created_at": batch_job.created_at.isoformat(),
            "metadata": batch_job.metadata,
            "finished": batch_job.finished,
            "postprocessing_chain": chain_b64,
        }
        path = self._job_path(batch_job.id)
        path.write_text(json.dumps(data, indent=2))

    async def get(self, job_id: str) -> BatchJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None

        job_data = json.loads(path.read_text())

        # Deserialize the chain using cloudpickle
        chain_b64 = job_data["postprocessing_chain"]
        chain_bytes = base64.b64decode(chain_b64)
        postprocessing_chain = cloudpickle.loads(chain_bytes)

        return BatchJob(
            id=job_data["id"],
            provider=job_data["provider"],
            created_at=datetime.fromisoformat(job_data["created_at"]),
            metadata=job_data.get("metadata", {}),
            finished=job_data.get("finished", False),
            postprocessing_chain=postprocessing_chain,
        )

    async def list(self, pending: bool = True) -> list[BatchJob]:
        keep_all_results = not pending
        results = []
        for path in self.storage_dir.glob("*.json"):
            job_id = path.stem
            item = await self.get(job_id)
            if item is not None:
                if keep_all_results or not item.finished:
                    results.append(item)
        return results

    async def delete(self, job_id: str) -> None:
        path = self._job_path(job_id)
        if path.exists():
            path.unlink()
