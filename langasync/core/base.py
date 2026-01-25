"""Base classes and interfaces for providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseProvider(ABC):
    """Base class for all provider implementations."""

    def __init__(self, api_key: Optional[str] = None, **config: Any):
        """Initialize the provider.

        Args:
            api_key: API key for authentication
            **config: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection and cleanup resources."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
