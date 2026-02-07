"""Tests for BatchChainWrapper and batch_chain factory."""

import pytest

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langasync.core.batch_chain import BatchChainWrapper, batch_chain
from langasync.core.batch_handle import BatchJobHandle
from langasync.core.batch_job_repository import FileSystemBatchJobRepository
from .conftest import MockChatModel


class TestBatchChainWrapperInit:
    """Tests for BatchChainWrapper initialization."""

    def test_init_extracts_model_from_chain(self, test_settings, chat_model: MockChatModel):
        """BatchChainWrapper extracts the model from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model | StrOutputParser()

        wrapper = BatchChainWrapper(chain, test_settings)

        assert wrapper.model is chat_model

    def test_init_extracts_preprocessing_from_chain(self, test_settings, chat_model: MockChatModel):
        """BatchChainWrapper extracts preprocessing steps from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model | StrOutputParser()

        wrapper = BatchChainWrapper(chain, test_settings)

        assert wrapper.preprocessing_chain is prompt

    def test_init_extracts_postprocessing_from_chain(
        self, test_settings, chat_model: MockChatModel
    ):
        """BatchChainWrapper extracts postprocessing steps from the chain."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        parser = StrOutputParser()
        chain = prompt | chat_model | parser

        wrapper = BatchChainWrapper(chain, test_settings)

        assert wrapper.postprocessing_chain is parser

    def test_init_with_no_model_chain(self, test_settings):
        """BatchChainWrapper handles chains with no model."""
        chain = RunnableLambda(lambda x: x.upper())

        wrapper = BatchChainWrapper(chain, test_settings)

        assert wrapper.model is None
        assert wrapper.preprocessing_chain is chain
        assert isinstance(wrapper.postprocessing_chain, RunnablePassthrough)

    def test_init_stores_batch_job_service(self, test_settings):
        """BatchChainWrapper creates and stores a BatchJobService."""
        chain = RunnablePassthrough()

        wrapper = BatchChainWrapper(chain, test_settings)

        assert wrapper.batch_job_service is not None


class TestBatchChainWrapperSubmit:
    """Tests for BatchChainWrapper.submit()."""

    @pytest.mark.asyncio
    async def test_submit_returns_batch_job_handle(self, test_settings):
        """submit() returns a BatchJobHandle."""
        chain = RunnableLambda(lambda x: x.upper())
        wrapper = BatchChainWrapper(chain, test_settings)

        handle = await wrapper.submit(["hello", "world"])

        assert isinstance(handle, BatchJobHandle)

    @pytest.mark.asyncio
    async def test_submit_saves_job_to_repository(self, test_settings):
        """submit() saves the job to the repository."""
        chain = RunnableLambda(lambda x: x.upper())
        wrapper = BatchChainWrapper(chain, test_settings)

        handle = await wrapper.submit(["hello"])

        repository = FileSystemBatchJobRepository(test_settings)
        saved_job = await repository.get(handle.job_id)
        assert saved_job is not None
        assert saved_job.id == handle.job_id

    @pytest.mark.asyncio
    async def test_submit_processes_inputs_through_preprocessing(self, test_settings):
        """submit() runs inputs through the preprocessing chain."""
        preprocessing = RunnableLambda(lambda x: f"pre:{x}")
        postprocessing = RunnableLambda(lambda x: f"post:{x}")
        chain = preprocessing | postprocessing
        wrapper = BatchChainWrapper(chain, test_settings)

        handle = await wrapper.submit(["a", "b"])
        result = await handle.get_results()

        assert result.results is not None
        assert "post:pre:a" in result.results
        assert "post:pre:b" in result.results

    @pytest.mark.asyncio
    async def test_submit_multiple_times_creates_separate_jobs(self, test_settings):
        """Each submit() call creates a separate job."""
        chain = RunnablePassthrough()
        wrapper = BatchChainWrapper(chain, test_settings)

        handle1 = await wrapper.submit(["a"])
        handle2 = await wrapper.submit(["b"])

        assert handle1.job_id != handle2.job_id

        repository = FileSystemBatchJobRepository(test_settings)
        jobs = await repository.list(pending=False)
        assert len(jobs) == 2


class TestBatchChainFactory:
    """Tests for batch_chain() factory function."""

    def test_batch_chain_returns_wrapper(self, test_settings):
        """batch_chain() returns a BatchChainWrapper."""
        chain = RunnablePassthrough()

        wrapper = batch_chain(chain, test_settings)

        assert isinstance(wrapper, BatchChainWrapper)

    def test_batch_chain_passes_chain_to_wrapper(self, test_settings, chat_model: MockChatModel):
        """batch_chain() passes the chain to the wrapper."""
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | chat_model

        wrapper = batch_chain(chain, test_settings)

        assert wrapper.model is chat_model
        assert wrapper.preprocessing_chain is prompt

    def test_batch_chain_passes_settings_to_wrapper(self, test_settings):
        """batch_chain() passes settings to the wrapper."""
        chain = RunnablePassthrough()

        wrapper = batch_chain(chain, test_settings)

        assert wrapper.batch_job_service is not None

    @pytest.mark.asyncio
    async def test_batch_chain_wrapper_can_submit(self, test_settings):
        """Wrapper from batch_chain() can submit jobs."""
        chain = RunnableLambda(lambda x: x * 2)
        wrapper = batch_chain(chain, test_settings)

        handle = await wrapper.submit([1, 2, 3])
        result = await handle.get_results()

        assert result.results is not None
        assert 2 in result.results
        assert 4 in result.results
        assert 6 in result.results
