"""Tests for BatchChainWrapper and batch_chain factory."""

import pytest
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_chain import BatchChainWrapper, batch_chain
from langasync.core.batch_service import BatchJobService
from langasync.core.batch_job_repository import FileSystemBatchJobRepository
from .conftest import MockChatModel


@pytest.fixture
def repository(tmp_path: Path) -> FileSystemBatchJobRepository:
    """Create a repository using a temporary directory."""
    return FileSystemBatchJobRepository(tmp_path)


class TestBatchChainWrapperInit:
    """Tests for BatchChainWrapper initialization."""

    def test_init_extracts_model_from_chain(
        self, repository: FileSystemBatchJobRepository, chat_model: MockChatModel
    ):
        """BatchChainWrapper extracts the model from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model | StrOutputParser()

        wrapper = BatchChainWrapper(chain, repository)

        assert wrapper.model is chat_model

    def test_init_extracts_preprocessing_from_chain(
        self, repository: FileSystemBatchJobRepository, chat_model: MockChatModel
    ):
        """BatchChainWrapper extracts preprocessing steps from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model | StrOutputParser()

        wrapper = BatchChainWrapper(chain, repository)

        assert wrapper.preprocessing_chain is prompt

    def test_init_extracts_postprocessing_from_chain(
        self, repository: FileSystemBatchJobRepository, chat_model: MockChatModel
    ):
        """BatchChainWrapper extracts postprocessing steps from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        parser = StrOutputParser()
        chain = prompt | chat_model | parser

        wrapper = BatchChainWrapper(chain, repository)

        assert wrapper.postprocessing_chain is parser

    def test_init_with_no_model_chain(self, repository: FileSystemBatchJobRepository):
        """BatchChainWrapper handles chains with no model."""
        chain = RunnableLambda(lambda x: x.upper())

        wrapper = BatchChainWrapper(chain, repository)

        assert wrapper.model is None
        assert wrapper.preprocessing_chain is chain
        assert isinstance(wrapper.postprocessing_chain, RunnablePassthrough)

    def test_init_stores_repository(self, repository: FileSystemBatchJobRepository):
        """BatchChainWrapper stores the repository."""
        chain = RunnablePassthrough()

        wrapper = BatchChainWrapper(chain, repository)

        assert wrapper.repository is repository


class TestBatchChainWrapperSubmit:
    """Tests for BatchChainWrapper.submit()."""

    @pytest.mark.asyncio
    async def test_submit_returns_batch_job_service(self, repository: FileSystemBatchJobRepository):
        """submit() returns a BatchJobService."""
        chain = RunnableLambda(lambda x: x.upper())
        wrapper = BatchChainWrapper(chain, repository)

        service = await wrapper.submit(["hello", "world"])

        assert isinstance(service, BatchJobService)

    @pytest.mark.asyncio
    async def test_submit_saves_job_to_repository(self, repository: FileSystemBatchJobRepository):
        """submit() saves the job to the repository."""
        chain = RunnableLambda(lambda x: x.upper())
        wrapper = BatchChainWrapper(chain, repository)

        service = await wrapper.submit(["hello"])

        saved_job = await repository.get(service.job_id)
        assert saved_job is not None
        assert saved_job.id == service.job_id

    @pytest.mark.asyncio
    async def test_submit_processes_inputs_through_preprocessing(
        self, repository: FileSystemBatchJobRepository
    ):
        """submit() runs inputs through the preprocessing chain."""
        preprocessing = RunnableLambda(lambda x: f"pre:{x}")
        postprocessing = RunnableLambda(lambda x: f"post:{x}")
        chain = preprocessing | postprocessing
        wrapper = BatchChainWrapper(chain, repository)

        service = await wrapper.submit(["a", "b"])
        result = await service.get_results()

        assert result.results is not None
        assert "post:pre:a" in result.results
        assert "post:pre:b" in result.results

    @pytest.mark.asyncio
    async def test_submit_multiple_times_creates_separate_jobs(
        self, repository: FileSystemBatchJobRepository
    ):
        """Each submit() call creates a separate job."""
        chain = RunnablePassthrough()
        wrapper = BatchChainWrapper(chain, repository)

        service1 = await wrapper.submit(["a"])
        service2 = await wrapper.submit(["b"])

        assert service1.job_id != service2.job_id

        jobs = await repository.list(pending=False)
        assert len(jobs) == 2


class TestBatchChainFactory:
    """Tests for batch_chain() factory function."""

    def test_batch_chain_returns_wrapper(self, repository: FileSystemBatchJobRepository):
        """batch_chain() returns a BatchChainWrapper."""
        chain = RunnablePassthrough()

        wrapper = batch_chain(chain, repository)

        assert isinstance(wrapper, BatchChainWrapper)

    def test_batch_chain_passes_chain_to_wrapper(
        self, repository: FileSystemBatchJobRepository, chat_model: MockChatModel
    ):
        """batch_chain() passes the chain to the wrapper."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model

        wrapper = batch_chain(chain, repository)

        assert wrapper.model is chat_model
        assert wrapper.preprocessing_chain is prompt

    def test_batch_chain_passes_repository_to_wrapper(
        self, repository: FileSystemBatchJobRepository
    ):
        """batch_chain() passes the repository to the wrapper."""
        chain = RunnablePassthrough()

        wrapper = batch_chain(chain, repository)

        assert wrapper.repository is repository

    @pytest.mark.asyncio
    async def test_batch_chain_wrapper_can_submit(self, repository: FileSystemBatchJobRepository):
        """Wrapper from batch_chain() can submit jobs."""
        chain = RunnableLambda(lambda x: x * 2)
        wrapper = batch_chain(chain, repository)

        service = await wrapper.submit([1, 2, 3])
        result = await service.get_results()

        assert result.results is not None
        assert 2 in result.results
        assert 4 in result.results
        assert 6 in result.results
