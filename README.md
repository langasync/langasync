# langasync

Access provider batch APIs through LangChain workflows. Get 50% cost savings on large-scale processing with minimal code changes.

## Why langasync?

Provider batch APIs (OpenAI Batch API, Anthropic Message Batches) offer 50% cost savings for non-urgent processing. langasync makes them accessible for LangChain chains:

```python
# Wrap your existing chain
batch_wrapper = batch_chain(chain, repository)

# Submit batch job
batch_job_service = await batch_wrapper.submit(inputs)

# Get results when ready
result = await batch_job_service.get_results()
```

No rewrite needed - same chain, just wrapped.

## Installation

```bash
pip install -e .
```

With dev dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langasync.core import batch_chain, FileSystemBatchJobRepository, BatchStatus

# Your existing chain
model = ChatOpenAI()
chain = prompt | model | parser

# Create repository for job persistence
repository = FileSystemBatchJobRepository("./batch_jobs")

# Wrap and submit
batch_wrapper = batch_chain(chain, repository)
batch_job_service = await batch_wrapper.submit([
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "Robotics"}
])

# Check results
result = await batch_job_service.get_results()
if result.status_info.status == BatchStatus.COMPLETED:
    print(f"Got {len(result.results)} results")
```

## Job Persistence

Batch jobs can take up to 24 hours. Jobs are automatically persisted, so you can resume after process restart:

```python
# Later, poll all pending jobs
from langasync.core import BatchPoller

repository = FileSystemBatchJobRepository("./batch_jobs")
poller = BatchPoller(repository)

async for result in poller.wait_all():
    print(f"Job {result.job_id}: {len(result.results)} results")
```

## Supported Providers

- OpenAI Batch API
- Anthropic Message Batches

## Running Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE) for terms.
