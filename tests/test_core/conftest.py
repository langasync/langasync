"""Shared test fixtures for core module tests."""

import os
import shutil
from pathlib import Path

import pytest

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever


class MockChatModel(BaseChatModel):
    """A minimal mock chat model for testing."""

    model_name: str = "mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="mock response"))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat"


class MockLLM(BaseLLM):
    """A minimal mock LLM for testing."""

    model_name: str = "mock-llm"

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        **kwargs,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text="mock response")]])

    @property
    def _llm_type(self) -> str:
        return "mock-llm"


class AnotherMockChatModel(BaseChatModel):
    """A second mock chat model for testing multiple models."""

    model_name: str = "another-mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="another mock"))])

    @property
    def _llm_type(self) -> str:
        return "another-mock-chat"


class MockRetriever(BaseRetriever):
    """A minimal mock retriever for testing."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return [Document(page_content=f"Mock document for: {query}")]


@pytest.fixture
def chat_model() -> MockChatModel:
    """Return a mock chat model."""
    return MockChatModel()


@pytest.fixture
def llm_model() -> MockLLM:
    """Return a mock LLM."""
    return MockLLM()


@pytest.fixture
def another_model() -> AnotherMockChatModel:
    """Return a second mock model for multi-model tests."""
    return AnotherMockChatModel()


@pytest.fixture
def chat_prompt() -> ChatPromptTemplate:
    """Return a simple chat prompt template."""
    return ChatPromptTemplate.from_messages([("user", "{input}")])


@pytest.fixture
def string_prompt() -> PromptTemplate:
    """Return a simple string prompt template."""
    return PromptTemplate.from_template("{input}")


@pytest.fixture
def str_parser() -> StrOutputParser:
    """Return a string output parser."""
    return StrOutputParser()


@pytest.fixture
def mock_retriever() -> MockRetriever:
    """Return a mock retriever."""
    return MockRetriever()


@pytest.fixture(autouse=True)
def clean_test_storage():
    """Clean up test storage directory before and after each test."""
    storage_dir = Path(os.environ.get("LANGASYNC_STORAGE_DIR", ".test_storage"))
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    yield
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
