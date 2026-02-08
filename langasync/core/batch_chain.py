from langchain_core.runnables import Runnable

from langasync.core.batch_service import BatchJobService
from langasync.core.batch_handle import BatchJobHandle
from langasync.core.get_parts_from_chain import get_parts_from_chain
from langasync.settings import LangasyncSettings, langasync_settings


class BatchChainWrapper:
    def __init__(
        self,
        chain: Runnable,
        settings: LangasyncSettings,
    ) -> None:
        parts = get_parts_from_chain(chain)

        self.model = parts.model
        self.model_bindings = parts.model_bindings
        self.preprocessing_chain = parts.preprocessing
        self.postprocessing_chain = parts.postprocessing
        self.batch_job_service = BatchJobService(settings)

    async def submit(self, inputs: list) -> BatchJobHandle:
        if not inputs:
            raise ValueError("inputs must be a non-empty list")
        return await self.batch_job_service.create(
            inputs,
            self.model,
            self.preprocessing_chain,
            self.postprocessing_chain,
            self.model_bindings,
        )


def batch_chain(
    chain: Runnable, settings: LangasyncSettings = langasync_settings
) -> BatchChainWrapper:
    return BatchChainWrapper(chain, settings)
