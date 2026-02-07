"""Integration tests that run against real provider APIs.

These tests make real API calls and cost money. Run manually or in nightly CI.

Usage:
    pytest -m integration tests/integration/

Requires API keys in environment:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
"""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langasync.core import FileSystemBatchJobRepository, batch_chain, BatchPoller
from langasync.providers.interface import BatchStatus


class CountryInfo(BaseModel):
    """Structured information about a country."""

    capital: str = Field(description="The capital city")
    population_millions: float = Field(description="Approximate population in millions")
    continent: str = Field(description="The continent the country is in")
    fun_fact: str = Field(description="An interesting fact about this country")


@pytest.fixture
def parser():
    return PydanticOutputParser(pydantic_object=CountryInfo)


@pytest.fixture
def prompt(parser):
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides country information."),
            ("user", "Give me information about {country}.\n\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI Batch API."""

    @pytest.fixture
    def openai_model(self):
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        return ChatOpenAI(model="gpt-4o-mini")

    async def test_batch_submit_and_poll(self, prompt, parser, openai_model):
        """Test full batch flow: submit, poll, get results."""
        with tempfile.TemporaryDirectory() as jobs_dir:
            chain = prompt | openai_model | parser
            repository = FileSystemBatchJobRepository(Path(jobs_dir))
            batch_wrapper = batch_chain(chain, repository)

            # Submit with single input for speed/cost
            batch_job = await batch_wrapper.submit([{"country": "France"}])
            assert batch_job.job_id is not None

            # Poll until complete
            poller = BatchPoller(repository)
            async for result in poller.wait_all():
                assert result.status_info.status in (
                    BatchStatus.COMPLETED,
                    BatchStatus.FAILED,
                    BatchStatus.EXPIRED,
                )

                if result.status_info.status == BatchStatus.COMPLETED:
                    assert len(result.results) == 1
                    country_info = result.results[0]
                    assert isinstance(country_info, CountryInfo)
                    assert country_info.capital == "Paris"
                    assert country_info.continent == "Europe"


@pytest.mark.integration
class TestAnthropicIntegration:
    """Integration tests for Anthropic Batch API."""

    @pytest.fixture
    def anthropic_model(self):
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(model="claude-sonnet-4-5-20250929")

    async def test_batch_submit_and_poll(self, prompt, parser, anthropic_model):
        """Test full batch flow: submit, poll, get results."""
        with tempfile.TemporaryDirectory() as jobs_dir:
            chain = prompt | anthropic_model | parser
            repository = FileSystemBatchJobRepository(Path(jobs_dir))
            batch_wrapper = batch_chain(chain, repository)

            # Submit with single input for speed/cost
            batch_job = await batch_wrapper.submit([{"country": "France"}])
            assert batch_job.job_id is not None

            # Poll until complete
            poller = BatchPoller(repository)
            async for result in poller.wait_all():
                assert result.status_info.status in (
                    BatchStatus.COMPLETED,
                    BatchStatus.FAILED,
                    BatchStatus.EXPIRED,
                )

                if result.status_info.status == BatchStatus.COMPLETED:
                    assert len(result.results) == 1
                    country_info = result.results[0]
                    assert isinstance(country_info, CountryInfo)
                    assert country_info.capital == "Paris"
                    assert country_info.continent == "Europe"
