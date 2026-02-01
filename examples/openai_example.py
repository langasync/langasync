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

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langasync.core import FileSystemBatchJobRepository, batch_chain, BatchPoller
from langasync.core.batch_api import BatchStatus

# Persistent storage directory
JOBS_DIR = Path(__file__).parent / ".batch_jobs"


async def run():
    """Submit a new batch job."""
    prompt = ChatPromptTemplate.from_template(
        "What is the capital of {country}? Answer in one word."
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    chain = prompt | model | parser
    repository = FileSystemBatchJobRepository(JOBS_DIR)
    batch_wrapper = batch_chain(chain, repository)

    inputs = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"},
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
            print(f"\nJob {result.job_id} completed: {result.results}")
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
