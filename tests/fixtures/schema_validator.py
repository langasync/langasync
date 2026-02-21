"""Schema validation using official SDK Pydantic models.

Validates factory outputs against the SDK's Pydantic models. When providers
update their SDKs, validation will fail automatically if our factories
produce incompatible structures.

Usage:
    from tests.fixtures.schema_validator import validate_openai_batch, validate_anthropic_batch

    # Validates by constructing SDK model - fails if schema mismatches
    validate_openai_batch(openai_batch_response(...))
    validate_anthropic_batch(anthropic_batch_response(...))
"""

from typing import Any

from openai.types import Batch as OpenAIBatch, FileObject as OpenAIFile, BatchError, ErrorObject
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from anthropic.types.messages import (
    MessageBatch as AnthropicBatch,
    MessageBatchIndividualResponse as AnthropicResultLine,
)
from google.genai.types import (
    Candidate as GeminiCandidate,
    GenerateContentResponse as GeminiResponse,
    InlinedResponse as GeminiInlinedResponse,
    JobError as GeminiJobError,
    ProjectOperation as GeminiOperation,
)


class SchemaValidationError(Exception):
    """Raised when a response doesn't match the SDK schema."""

    pass


def validate_openai_batch(response: dict[str, Any]) -> OpenAIBatch:
    """Validate response against OpenAI Batch model."""
    try:
        return OpenAIBatch.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"OpenAI Batch validation failed: {e}") from e


def validate_openai_file(response: dict[str, Any]) -> OpenAIFile:
    """Validate response against OpenAI FileObject model."""
    try:
        return OpenAIFile.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"OpenAI FileObject validation failed: {e}") from e


def validate_openai_chat_completion(response: dict[str, Any]) -> OpenAIChatCompletion:
    """Validate response against OpenAI ChatCompletion model."""
    try:
        return OpenAIChatCompletion.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"OpenAI ChatCompletion validation failed: {e}") from e


def validate_openai_batch_error(response: dict[str, Any]) -> BatchError:
    """Validate response against OpenAI BatchError model."""
    try:
        return BatchError.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"OpenAI BatchError validation failed: {e}") from e


def validate_openai_error_object(response: dict[str, Any]) -> ErrorObject:
    """Validate response against OpenAI ErrorObject model."""
    try:
        return ErrorObject.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"OpenAI ErrorObject validation failed: {e}") from e


def validate_anthropic_batch(response: dict[str, Any]) -> AnthropicBatch:
    """Validate response against Anthropic MessageBatch model."""
    try:
        return AnthropicBatch.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Anthropic MessageBatch validation failed: {e}") from e


def validate_anthropic_result_line(response: dict[str, Any]) -> AnthropicResultLine:
    """Validate response against Anthropic MessageBatchIndividualResponse model."""
    try:
        return AnthropicResultLine.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Anthropic result line validation failed: {e}") from e


def validate_gemini_operation(response: dict[str, Any]) -> GeminiOperation:
    """Validate response against Google Gemini ProjectOperation model.

    Validates the Operation/LRO wrapper (name, metadata, done).
    The 'response' field (inlined results) is excluded since ProjectOperation
    doesn't model it — inline responses are validated separately via
    validate_gemini_response.
    """
    try:
        # Exclude 'response' field — ProjectOperation doesn't have it
        filtered = {k: v for k, v in response.items() if k != "response"}
        return GeminiOperation.model_validate(filtered)
    except Exception as e:
        raise SchemaValidationError(f"Gemini Operation validation failed: {e}") from e


def validate_gemini_candidate(response: dict[str, Any]) -> GeminiCandidate:
    """Validate response against Google Gemini Candidate model."""
    try:
        return GeminiCandidate.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Gemini Candidate validation failed: {e}") from e


def validate_gemini_response(response: dict[str, Any]) -> GeminiResponse:
    """Validate response against Google Gemini GenerateContentResponse model."""
    try:
        return GeminiResponse.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Gemini GenerateContentResponse validation failed: {e}") from e


def validate_gemini_inline_response(response: dict[str, Any]) -> GeminiInlinedResponse:
    """Validate response against Google Gemini InlinedResponse model."""
    try:
        return GeminiInlinedResponse.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Gemini InlinedResponse validation failed: {e}") from e


def validate_gemini_error(response: dict[str, Any]) -> GeminiJobError:
    """Validate response against Google Gemini JobError model."""
    try:
        return GeminiJobError.model_validate(response)
    except Exception as e:
        raise SchemaValidationError(f"Gemini JobError validation failed: {e}") from e
