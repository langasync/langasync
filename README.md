<p align="center">
  <a href="https://github.com/256Bab/langasync">
    <img alt="langasync" src="langasync.png" width="400">
  </a>
</p>

<p align="center">
  <strong>50% cost savings on LLM APIs with zero code changes.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/langasync/"><img alt="PyPI" src="https://img.shields.io/pypi/v/langasync?color=blue"></a>
  <a href="https://pypi.org/project/langasync/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/langasync"></a>
  <a href="https://github.com/256Bab/langasync/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-BSL--1.1-blue"></a>
  <a href="https://github.com/256Bab/langasync/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/256Bab/langasync/actions/workflows/test.yml/badge.svg"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#supported-providers">Providers</a> •
  <a href="#contributing">Contributing</a>
</p>

---

**langasync** lets you use provider batch APIs (OpenAI, Anthropic) with your existing LangChain chains. Wrap your chain, submit inputs, get results at half the cost.

```python
from langasync.core import batch_chain, FileSystemBatchJobRepository

# Your existing chain — no changes needed
chain = prompt | model | parser

# Wrap for batch processing
batch_wrapper = batch_chain(chain, FileSystemBatchJobRepository("./jobs"))

# Submit and retrieve results
job = await batch_wrapper.submit(inputs)
results = await job.get_results()
```

## Why langasync?

Provider batch APIs offer **50% cost savings** on tokens for workloads that can tolerate 24-hour turnaround. But they require completely different code patterns — file uploads, polling, result parsing.

langasync abstracts all of that:

| Without langasync | With langasync |
|-------------------|----------------|
| Rewrite chains as batch requests | Same chain, just wrapped |
| Manage file uploads (OpenAI) | Automatic |
| Build custom polling logic | Built-in `BatchPoller` |
| Parse provider-specific responses | Unified `BatchItem` |
| Handle partial failures manually | Automatic success/error separation |

## Quick Start

### Installation

```bash
pip install langasync
```

### Basic Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langasync.core import batch_chain, FileSystemBatchJobRepository, BatchStatus

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Explain {topic} in one paragraph.")
])
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model

repository = FileSystemBatchJobRepository("./batch_jobs")
batch_wrapper = batch_chain(chain, repository)

async def main():
    job = await batch_wrapper.submit([
        {"topic": "quantum computing"},
        {"topic": "machine learning"},
        {"topic": "blockchain"},
    ])

    result = await job.get_results()

    if result.status_info.status == BatchStatus.COMPLETED:
        for r in result.results:
            print(r.content)

asyncio.run(main())
```

## Features

### Drop-in Batch Support

Wrap any LangChain chain with `batch_chain()`. Prompts, models, parsers — all work automatically.

```python
chain = prompt | model | parser
batch_wrapper = batch_chain(chain, repository)
```

### Structured Output

Full support for Pydantic output parsers and schemas:

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class Analysis(BaseModel):
    sentiment: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=Analysis)
chain = prompt | model | parser
```

### Tool Calling

`.bind_tools()` works out of the box:

```python
model = ChatOpenAI().bind_tools([my_tool])
chain = prompt | model
```

### Job Persistence

Batch jobs can take up to 24 hours. Jobs persist automatically — resume after process restart:

```python
from langasync.core import BatchPoller

# Later, in a new process
poller = BatchPoller(repository)

async for result in poller.wait_all():
    print(f"Job {result.job_id}: {result.status_info.status}")
```

### Partial Failure Handling

Get successful results even when some requests fail:

```python
result = await job.get_results()

for r in result.results:
    if r.success:
        print(r.content)
    else:
        print(f"Failed: {r.error}")
```

## Supported Providers

| Provider | Status | Batch API | Savings |
|----------|--------|-----------|---------|
| **OpenAI** | ✅ Supported | [Batch API](https://platform.openai.com/docs/guides/batch) | 50% |
| **Anthropic** | ✅ Supported | [Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) | 50% |
| Google Vertex AI | 🔜 Planned | — | — |
| Azure OpenAI | 🔜 Planned | — | — |

## Documentation

### API Reference

| Function / Class | Description |
|------------------|-------------|
| `batch_chain(chain, repository)` | Wrap a LangChain chain for batch processing |
| `BatchPoller(repository)` | Poll pending jobs and retrieve results |
| `FileSystemBatchJobRepository(path)` | Persist jobs to local filesystem |
| `BatchJobService` | Returned by `submit()` — provides `get_status()`, `get_results()`, `cancel()` |
| `BatchStatus` | Enum: `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, `EXPIRED` |

### Configuration

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Examples

See [examples/](examples/) for complete working examples:

```bash
# Submit a batch job
python examples/openai_example.py run

# Fetch results (can run later, after restart)
python examples/openai_example.py fetch
```

## Development

### Setup

```bash
git clone https://github.com/256Bab/langasync.git
cd langasync
pip install -e ".[dev]"
```

### Running Tests

```bash
# Unit tests (mocked, fast)
pytest tests

# Integration tests (requires API keys)
pytest tests/integration -o "addopts="
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a pull request

## Community

- [GitHub Issues](https://github.com/256Bab/langasync/issues) — Bug reports and feature requests
- [GitHub Discussions](https://github.com/256Bab/langasync/discussions) — Questions and ideas

## License

langasync is licensed under the [Business Source License 1.1](LICENSE). You can use it freely for any purpose, including commercial and production use, as long as you don't offer it as a competing hosted service. On February 6, 2030, the license converts to Apache 2.0.
