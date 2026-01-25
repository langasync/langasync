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
