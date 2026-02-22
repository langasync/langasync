# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-02-22

### Added
- AWS Bedrock Batch Inference provider (`BedrockProviderJobAdapter`) with S3-based input/output
- Google Gemini Batch API provider (`GeminiProviderJobAdapter`)
- Bedrock model provider strategy pattern (`BedrockModelProvider`) with Anthropic support
- Auto-detection of Bedrock provider from model ID (e.g. `anthropic.claude-3-sonnet-...`)
- Region prefix auto-mapping for Bedrock cross-region inference profiles
- OpenAI-to-Anthropic tool format conversion for `ChatBedrockConverse.bind_tools()`
- Gemini thinking/reasoning support
- Bedrock and Gemini examples

### Changed
- Cancel now returns quickly once OpenAI confirms cancellation is in progress
- Updated `langchain-aws` dependency to `>=1.0,<2.0` for `langchain-core` 1.x compatibility

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
