"""Pass-through adapter for chains without a model."""

from abc import ABC, abstractmethod
from typing import Callable
from datetime import datetime
from pathlib import Path

import cloudpickle  # type: ignore[import-untyped]
from langchain_core.language_models import LanguageModelInput

from langasync.exceptions import provider_error_handling
from langasync.settings import langasync_settings, LangasyncSettings
from langasync.utils import generate_uuid
from langasync.providers.interface import (
    ProviderJobAdapterInterface,
    ProviderJob,
    BatchItem,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
    Provider,
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


def metadata_persistence_factory(settings: LangasyncSettings) -> NoModelApiPersistence:
    if settings.base_storage_path:
        metadata_path = Path(settings.base_storage_path) / "no_model_metadata"
        return FileSystemNoModelApiPersistence(metadata_path)
    else:
        raise NotImplementedError()


class NoModelProviderJobAdapter(ProviderJobAdapterInterface):

    def __init__(
        self,
        settings: LangasyncSettings,
        _metadata_persistence_factory: Callable[
            [LangasyncSettings], NoModelApiPersistence
        ] = metadata_persistence_factory,
    ):
        self._persistence = _metadata_persistence_factory(settings)

    @provider_error_handling
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
        model_bindings: dict | None = None,
    ) -> ProviderJob:
        batch_id = f"no-model-{generate_uuid()}"
        batch_api_job = ProviderJob(
            id=batch_id,
            provider=Provider.NONE,
            created_at=datetime.now(),
        )
        await self._persistence.save(batch_id, inputs)
        return batch_api_job

    @provider_error_handling
    async def get_status(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        inputs = await self._persistence.load(batch_api_job.id)
        return BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total=len(inputs),
            completed=len(inputs),
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> list[ProviderJob]:
        raise NotImplementedError("list_batches not implemented on NoModelProviderJobAdapter")

    @provider_error_handling
    async def get_results(self, batch_api_job: ProviderJob) -> list[BatchItem]:
        inputs = await self._persistence.load(batch_api_job.id)
        return [
            BatchItem(
                custom_id=str(i),
                success=True,
                content=inp,
            )
            for i, inp in enumerate(inputs)
        ]

    @provider_error_handling
    async def cancel(self, batch_api_job: ProviderJob) -> BatchStatusInfo:
        inputs = await self._persistence.load(batch_api_job.id)
        return BatchStatusInfo(
            status=BatchStatus.CANCELLED,
            total=len(inputs),
            completed=len(inputs),
            failed=0,
        )
