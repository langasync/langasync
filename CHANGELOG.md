# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-02-07

### Added
- `batch_chain()` wrapper for submitting LangChain chains to provider batch APIs
- `BatchPoller` for polling pending jobs and retrieving results
- `LangasyncSettings` for centralized configuration (API keys, storage path, poll interval)
- OpenAI Batch API provider (`OpenAIProviderJobAdapter`)
- Anthropic Message Batches API provider (`AnthropicProviderJobAdapter`)
- Support for `.bind_tools()` and `.bind()` kwargs in batch requests
- Multimodal input support (images, PDFs, files)
- `FileSystemBatchJobRepository` for job metadata persistence across restarts
- `BatchJobHandle` with `get_results()` and `cancel()` methods
- Partial failure handling with per-item status indicators
- Cancel with polling until terminal state
- Examples for basic usage, multimodal inputs, and tool calling
