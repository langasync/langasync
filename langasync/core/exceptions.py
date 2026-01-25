"""Exception classes for langasync."""


class LangAsyncError(Exception):
    """Base exception for all langasync errors."""

    pass


class ProviderError(LangAsyncError):
    """Exception raised when a provider encounters an error."""

    pass


class ConfigurationError(LangAsyncError):
    """Exception raised when configuration is invalid."""

    pass


class AuthenticationError(ProviderError):
    """Exception raised when authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Exception raised when rate limit is exceeded."""

    pass


class UnsupportedChainError(LangAsyncError):
    """Exception raised when a chain contains unsupported components.

    This is raised when:
    - A chain contains multiple models
    - A chain contains retrievers
    - A chain contains other unsupported components
    """

    pass
