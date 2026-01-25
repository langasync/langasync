"""Pass-through adapter for chains without a model."""

import uuid
from datetime import datetime
from typing import Dict, List

from langchain_core.language_models import LanguageModelInput

from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchJob,
    BatchResponse,
    BatchStatus,
    BatchStatusInfo,
    LanguageModelType,
)


class NoModelBatchApiAdapter(BatchApiAdapterInterface):
    """Pass-through adapter for chains without a model. Results are immediate."""

    def __init__(self):
        self._batches: Dict[str, BatchJob] = {}
        self._inputs: Dict[str, List[LanguageModelInput]] = {}

    async def create_batch(
        self,
        inputs: List[LanguageModelInput],
        language_model: LanguageModelType,
    ) -> BatchJob:
        batch_id = f"no-model-{uuid.uuid4()}"
        batch_job = BatchJob(
            id=batch_id,
            provider="none",
            created_at=datetime.now(),
        )
        self._batches[batch_id] = batch_job
        self._inputs[batch_id] = inputs
        return batch_job

    async def get_status(self, batch_job: BatchJob) -> BatchStatusInfo:
        inputs = self._inputs.get(batch_job.id, [])
        return BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total=len(inputs),
            completed=len(inputs),
            failed=0,
        )

    async def list_batches(self, limit: int = 20) -> List[BatchJob]:
        return list(self._batches.values())[:limit]

    async def get_results(self, batch_job: BatchJob) -> List[BatchResponse]:
        inputs = self._inputs.get(batch_job.id, [])
        return [
            BatchResponse(
                custom_id=str(i),
                success=True,
                content=str(inp),
            )
            for i, inp in enumerate(inputs)
        ]

    async def cancel(self, batch_job: BatchJob) -> bool:
        return False  # Already completed, can't cancel
