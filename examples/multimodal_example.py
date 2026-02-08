"""Example script demonstrating Batch API usage with images and PDFs.

Usage:
    # Submit a batch job (default: openai)
    python examples/multimodal_example.py run
    python examples/multimodal_example.py run openai
    python examples/multimodal_example.py run anthropic

    # Fetch results (can be run later, even after process restart)
    python examples/multimodal_example.py fetch

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in environment or .env file.
"""

import asyncio
import base64
import sys

import httpx
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langasync import batch_chain, BatchPoller, LangasyncSettings, BatchStatus

load_dotenv()

settings = LangasyncSettings(base_storage_path="./examples/.batch_jobs")


def image_message(text: str, image_url: str) -> list:
    """Build a message list with an image URL."""
    return [
        SystemMessage(content="You are a helpful assistant that analyzes images and documents."),
        HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image", "url": image_url},
            ]
        ),
    ]


def pdf_message(text: str, pdf_url: str) -> list:
    """Build a message list with a base64-encoded PDF fetched from a URL."""
    pdf_bytes = httpx.get(
        pdf_url,
        headers={"User-Agent": "langasync/1.0 (https://github.com/langasync/langasync)"},
    ).content
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return [
        SystemMessage(content="You are a helpful assistant that analyzes images and documents."),
        HumanMessage(
            content=[
                {"type": "text", "text": text},
                {
                    "type": "file",
                    "base64": pdf_b64,
                    "mime_type": "application/pdf",
                    "filename": "document.pdf",
                },
            ]
        ),
    ]


async def run(model):
    """Submit a new batch job with image and PDF inputs."""
    # No prompt template — pass pre-built messages directly
    batch_wrapper = batch_chain(model, settings)

    inputs = [
        image_message(
            "Describe what you see in this image in one sentence.",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        ),
        image_message(
            "What colors are prominent in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",
        ),
        pdf_message(
            "Explain the contents of this file.",
            "https://upload.wikimedia.org/wikipedia/commons/1/13/Example.pdf",
        ),
    ]

    print(f"Submitting batch with {len(inputs)} inputs...")
    handle = await batch_wrapper.submit(inputs)
    print(f"Batch submitted: {handle.job_id}")
    print(f"Jobs stored in: {settings.base_storage_path}")
    print("\nRun 'python examples/multimodal_example.py fetch' to get results.")


async def fetch():
    """Fetch results from pending batch jobs."""
    poller = BatchPoller(settings)

    async for result in poller.wait_all():
        status = result.status_info.status
        if status == BatchStatus.COMPLETED:
            print(f"\nJob {result.job_id} completed:")
            for item in result.results:
                print(f"\n{item}")
        else:
            print(f"\nJob {result.job_id}: {status.value}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/multimodal_example.py [run|fetch] [openai|anthropic]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        provider = sys.argv[2] if len(sys.argv) > 2 else "openai"
        if provider == "openai":
            model = ChatOpenAI(model="gpt-4o-mini")
        elif provider == "anthropic":
            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        else:
            print(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
            sys.exit(1)
        asyncio.run(run(model))
    elif command == "fetch":
        asyncio.run(fetch())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
