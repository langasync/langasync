"""Example script demonstrating OpenAI Batch API usage.

Usage:
    # Submit a batch job
    python examples/openai_example.py run

    # Fetch results (can be run later, even after process restart)
    python examples/openai_example.py fetch

Requires OPENAI_API_KEY in environment or .env file.
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.getLogger("langasync").setLevel(logging.INFO)
logging.getLogger("langasync").addHandler(logging.StreamHandler())

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langasync.core import FileSystemBatchJobRepository, batch_chain, BatchPoller
from langasync.core.batch_api import BatchStatus

# Persistent storage directory
JOBS_DIR = Path(__file__).parent / ".batch_jobs"


class CountryInfo(BaseModel):
    """Structured information about a country."""

    capital: str = Field(description="The capital city")
    population_millions: float = Field(description="Approximate population in millions")
    continent: str = Field(description="The continent the country is in")
    fun_fact: str = Field(description="An interesting fact about this country")


async def run():
    """Submit a new batch job."""
    parser = PydanticOutputParser(pydantic_object=CountryInfo)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides country information."),
            ("user", "Give me information about {country}.\n\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    model = ChatOpenAI(model="gpt-4o-mini")

    chain = prompt | model | parser
    repository = FileSystemBatchJobRepository(JOBS_DIR)
    batch_wrapper = batch_chain(chain, repository)

    inputs = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"},
        {"country": "Kenya"},
        {"country": "Australia"},
    ]

    print(f"Submitting batch with {len(inputs)} inputs...")
    batch_job_service = await batch_wrapper.submit(inputs)
    print(f"Batch submitted: {batch_job_service.job_id}")
    print(f"Jobs stored in: {JOBS_DIR}")
    print("\nRun 'python examples/openai_example.py fetch' to get results.")


async def fetch():
    """Fetch results from pending batch jobs."""
    repository = FileSystemBatchJobRepository(JOBS_DIR)
    poller = BatchPoller(repository)

    async for result in poller.wait_all():
        status = result.status_info.status
        if status == BatchStatus.COMPLETED:
            print(f"\nJob {result.job_id} completed:")
            for result in result.results:
                print(result)
        else:
            print(f"\nJob {result.job_id}: {status.value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/openai_example.py [run|fetch]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        asyncio.run(run())
    elif command == "fetch":
        asyncio.run(fetch())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
