import functools


class LangAsyncError(Exception):
    """Base exception for all langasync errors."""

    pass


class FailedPreProcessingError(LangAsyncError):
    """Exception raised when the preprocessing chain fails."""

    pass


class FailedLLMOutputError(LangAsyncError):
    """Exception raised when the LLM returns an error or invalid response."""

    pass


class FailedPostProcessingError(LangAsyncError):
    """Exception raised when the output parser fails to process LLM output."""

    pass


class AuthenticationError(LangAsyncError):
    """Exception raised when authentication fails."""

    pass


class ApiTimeoutError(LangAsyncError):
    """Exception raised when provider timesout."""

    pass


class BatchProviderApiError(LangAsyncError):
    """Exception raised when provider llm api fails."""

    pass


class UnsupportedChainError(LangAsyncError):
    """Exception raised when a chain contains unsupported components.

    This is raised when:
    - A chain contains multiple models
    - A chain contains retrievers
    - A chain contains other unsupported components
    """

    pass


class UnsupportedProviderError(LangAsyncError):
    """Exception raised when a provider is not supported or not recognized."""

    pass


def error_handling(fn, default_exception_class=LangAsyncError):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except LangAsyncError:
            raise
        except Exception as e:
            raise default_exception_class(str(e)) from e

    return wrapper


def provider_error_handling(fn):
    return error_handling(fn, default_exception_class=BatchProviderApiError)
