"""Example script demonstrating Google Gemini Batch API usage.

Usage:
    # Submit a batch job
    python examples/gemini_example.py run

    # Fetch results (can be run later, even after process restart)
    python examples/gemini_example.py fetch

Requires GOOGLE_API_KEY in environment or .env file.
"""

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from langasync import batch_chain, BatchPoller, LangasyncSettings, BatchStatus

settings = LangasyncSettings(base_storage_path="./examples/.batch_jobs")


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

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    chain = prompt | model | parser
    batch_wrapper = batch_chain(chain, settings)

    inputs = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"},
        {"country": "Kenya"},
        {"country": "Australia"},
    ]

    print(f"Submitting batch with {len(inputs)} inputs...")
    handle = await batch_wrapper.submit(inputs)
    print(f"Batch submitted: {handle.job_id}")
    print(f"Jobs stored in: {settings.base_storage_path}")
    print("\nRun 'python examples/gemini_example.py fetch' to get results.")


async def fetch():
    """Fetch results from pending batch jobs."""
    poller = BatchPoller(settings)

    async for result in poller.wait_all():
        status = result.status_info.status
        if status == BatchStatus.COMPLETED:
            print(f"\nJob {result.job_id} completed:")
            for item in result.results:
                print(item)
        else:
            print(f"\nJob {result.job_id}: {status.value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/gemini_example.py [run|fetch]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        asyncio.run(run())
    elif command == "fetch":
        asyncio.run(fetch())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
