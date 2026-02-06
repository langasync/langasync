from langasync.core.batch_service import BatchJobService
from langasync.core.batch_job_repository import BatchJobRepository
from langasync.core.get_parts_from_chain import get_parts_from_chain


class BatchChainWrapper:
    def __init__(self, chain, repository: BatchJobRepository):
        parts = get_parts_from_chain(chain)

        self.model = parts.model
        self.model_bindings = parts.model_bindings
        self.preprocessing_chain = parts.preprocessing
        self.postprocessing_chain = parts.postprocessing
        self.repository = repository

    async def submit(self, inputs) -> BatchJobService:
        return await BatchJobService.create(
            inputs,
            self.model,
            self.preprocessing_chain,
            self.postprocessing_chain,
            self.repository,
            self.model_bindings,
        )


def batch_chain(chain, repository: BatchJobRepository) -> BatchChainWrapper:
    return BatchChainWrapper(chain, repository)
