"""Example script demonstrating Batch API usage with tools.

Usage:
    # Submit a batch job (default: anthropic)
    python examples/tools_example.py run
    python examples/tools_example.py run anthropic
    python examples/tools_example.py run openai
    python examples/tools_example.py run google
    python examples/tools_example.py run bedrock

    # Fetch results (can be run later, even after process restart)
    python examples/tools_example.py fetch

Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or AWS credentials in environment or .env file.
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langasync import batch_chain, BatchPoller, LangasyncSettings, BatchStatus

load_dotenv()


settings = LangasyncSettings(base_storage_path="./examples/.batch_jobs")


class GetWeather(BaseModel):
    """Get the current weather for a location."""

    location: str = Field(description="City and country, e.g. 'Paris, France'")
    unit: str = Field(default="celsius", description="Temperature unit: 'celsius' or 'fahrenheit'")


class SearchWeb(BaseModel):
    """Search the web for information."""

    query: str = Field(description="The search query")


async def run(model):
    """Submit a new batch job with tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use tools when appropriate."),
            ("user", "{question}"),
        ]
    )

    model_with_tools = model.bind_tools([GetWeather, SearchWeb])

    # No parser - we want the raw AIMessage with tool_calls
    chain = prompt | model_with_tools
    batch_wrapper = batch_chain(chain, settings)

    inputs = [
        {"question": "What's the weather like in Tokyo?"},
        {"question": "What's the weather in New York in fahrenheit?"},
        {"question": "Search for the latest news about AI"},
        {"question": "What's the temperature in London?"},
        {"question": "Find information about Python 3.12 features"},
    ]

    # Bedrock batch requires larger input counts
    if isinstance(model, ChatBedrockConverse):
        inputs = inputs * 20  # 100 inputs

    print(f"Submitting batch with {len(inputs)} inputs...")
    handle = await batch_wrapper.submit(inputs)
    print(f"Batch submitted: {handle.job_id}")
    print(f"Jobs stored in: {settings.base_storage_path}")
    print("\nRun 'python examples/tools_example.py fetch' to get results.")


async def fetch():
    """Fetch results from pending batch jobs."""
    poller = BatchPoller(settings)

    async for result in poller.wait_all():
        status = result.status_info.status
        if status == BatchStatus.COMPLETED:
            print(f"\nJob {result.job_id} completed:")
            for item in result.results:
                print(f"\n")
                print(item)
        else:
            print(f"\nJob {result.job_id}: {status.value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python examples/tools_example.py [run|fetch] [openai|anthropic|google|bedrock]"
        )
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        provider = sys.argv[2] if len(sys.argv) > 2 else "anthropic"
        if provider == "openai":
            model = ChatOpenAI(model="gpt-4o-mini")
        elif provider == "anthropic":
            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        elif provider == "google":
            model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        elif provider == "bedrock":
            model = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6")
        else:
            print(
                f"Unknown provider: {provider}. Use 'openai', 'anthropic', 'google', or 'bedrock'."
            )
            sys.exit(1)
        asyncio.run(run(model))
    elif command == "fetch":
        asyncio.run(fetch())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
