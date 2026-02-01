"""Pass-through adapter for chains without a model."""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cloudpickle
from langchain_core.language_models import LanguageModelInput

from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchApiJob,
    BatchResponse,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
)


class NoModelApiPersistence(ABC):
    """Interface for persisting NoModel batch inputs."""

    @abstractmethod
    async def save(self, batch_id: str, inputs: list[LanguageModelInput]) -> None:
        pass

    @abstractmethod
    async def load(self, batch_id: str) -> list[LanguageModelInput]:
        pass


class FileSystemNoModelApiPersistence(NoModelApiPersistence):
    """File-based persistence for NoModel batch inputs."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _inputs_path(self, batch_id: str) -> Path:
        return self.storage_dir / f"{batch_id}.inputs"

    async def save(self, batch_id: str, inputs: list[LanguageModelInput]) -> None:
        self._inputs_path(batch_id).write_bytes(cloudpickle.dumps(inputs))

    async def load(self, batch_id: str) -> list[LanguageModelInput]:
        path = self._inputs_path(batch_id)
        if path.exists():
            return cloudpickle.loads(path.read_bytes())
        return []


class NoModelBatchApiAdapter(BatchApiAdapterInterface):
    """Pass-through adapter for chains without a model. Results are immediate.

    Args:
        persistence: Optional persistence layer. If not provided, reads
                    LANGASYNC_STORAGE_DIR env var to create file-based persistence.
    """

    def __init__(self, persistence: NoModelApiPersistence | None = None):
        if persistence is None:
            storage_dir = Path(os.environ.get("LANGASYNC_STORAGE_DIR", ".langasync"))
            persistence = FileSystemNoModelApiPersistence(storage_dir / "no_model_metadata")
        self._persistence = persistence

    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
    ) -> BatchApiJob:
        batch_id = f"no-model-{uuid.uuid4()}"
        batch_api_job = BatchApiJob(
            id=batch_id,
            provider="none",
            created_at=datetime.now(),
        )
        await self._persistence.save(batch_id, inputs)
        return batch_api_job

    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        inputs = await self._persistence.load(batch_api_job.id)
        return BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total=len(inputs),
            completed=len(inputs),
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        return []

    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        inputs = await self._persistence.load(batch_api_job.id)
        return [
            BatchResponse(
                custom_id=str(i),
                success=True,
                content=inp,
            )
            for i, inp in enumerate(inputs)
        ]

    async def cancel(self, batch_api_job: BatchApiJob) -> bool:
        return False  # Already completed, can't cancel
