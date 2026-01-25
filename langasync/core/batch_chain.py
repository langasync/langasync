import asyncio

from langasync.core.batch_api import (
    BatchApiAdapterInterface,
    BatchJob,
    FINISHED_STATUSES,
    BatchResponse,
)
from langasync.core.get_parts_from_chain import get_parts_from_chain
from langasync.providers.no_provider import NoModelBatchApiAdapter
from langchain_core.runnables import Runnable
from typing import Callable, Any, List


class BatchJobService:
    def __init__(
        self,
        batch_job: BatchJob,
        batch_api_adapter: BatchApiAdapterInterface,
        batch_results_postprocessing: Callable[[list[BatchResponse]], list[Any]],
    ):
        self.batch_job = batch_job
        self.batch_api_adapter = batch_api_adapter
        self.batch_results_postprocessing = batch_results_postprocessing
        self.poll_frequency_seconds = 100

    async def get_results(self):
        batch_status = await self.batch_api_adapter.get_status(self.batch_job)
        if batch_status.status in FINISHED_STATUSES:
            results = await self.batch_api_adapter.get_results(self.batch_job)
            return self.batch_results_postprocessing(results)
        else:
            return None

    async def wait(self):
        results = None
        while results is None:
            results = await self.get_results()
            if results is None:
                await asyncio.sleep(self.poll_frequency_seconds)
        return results

    async def get_status(self):
        return await self.batch_api_adapter.get_status(self.batch_job)

    async def cancel(self):
        await self.batch_api_adapter.cancel(self.batch_job)
        return True


class BatchChainWrapper:
    def __init__(self, chain):
        # Decompose chain into parts
        parts = get_parts_from_chain(chain)

        self.model = parts.model
        self.preprocessing_chain = parts.preprocessing
        self.postprocessing_chain = parts.postprocessing

        # Get appropriate adapter based on model
        if self.model is not None:
            self.batch_api_adapter = self._get_batch_api_adapter(self.model)
        else:
            self.batch_api_adapter = NoModelBatchApiAdapter()

    def _make_postprocessor(self, chain: Runnable) -> Callable[[List[BatchResponse]], List[Any]]:
        """Wrap a postprocessing chain into a callable for BatchJobService."""

        async def postprocess(results: List[BatchResponse]) -> List[Any]:
            contents = [r.content for r in results]
            return await chain.abatch(contents)

        return postprocess

    def _get_batch_api_adapter(self, model) -> BatchApiAdapterInterface:
        """Get the appropriate batch API adapter for a model."""
        # TODO: Check model type/provider, return appropriate adapter
        # e.g., OpenAI models -> OpenAIBatchAdapter
        raise NotImplementedError

    async def submit(self, inputs):
        preprocessed_inputs = await self.preprocessing_chain.abatch(inputs)
        if self.batch_api_adapter is not None:
            batch_job = await self.batch_api_adapter.create_batch(preprocessed_inputs, self.model)
        postprocessor = self._make_postprocessor(self.postprocessing_chain)
        return BatchJobService(batch_job, self.batch_api_adapter, postprocessor)
