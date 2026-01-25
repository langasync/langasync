"""Batch API abstractions and interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from pydantic import BaseModel, Field


class NullLanguageModel(BaseModel):
    """Marker type indicating no language model is present in the chain."""

    pass


LanguageModelType = BaseLanguageModel | NullLanguageModel


class BatchStatus(str, Enum):
    """Status of a batch job across providers."""

    # Common states
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


FINISHED_STATUSES = {
    BatchStatus.COMPLETED,
    BatchStatus.FAILED,
    BatchStatus.CANCELLED,
    BatchStatus.EXPIRED,
}


class BatchResponse(BaseModel):
    """Response for a single request in a batch."""

    custom_id: str
    success: bool
    content: str | None = None
    error: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchStatusInfo(BaseModel):
    """Status information for a batch job, including request counts."""

    status: BatchStatus
    total: int
    completed: int
    failed: int


@dataclass
class BatchApiJob:
    """Simple data container for batch job information."""

    id: str
    provider: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchApiAdapterInterface(ABC):
    """Abstract interface for provider-specific batch API implementations."""

    @abstractmethod
    async def create_batch(
        self,
        inputs: list[LanguageModelInput],
        language_model: LanguageModelType,
    ) -> BatchApiJob:
        """Create a new batch job.

        Args:
            inputs: List of inputs (LangChain format), one per batch request.
                   Each input can be a string, list of messages, dict, etc.
            language_model: LangChain language model or NullLanguageModel if no model

        Returns:
            BatchJob handle for the created batch
        """
        pass

    @abstractmethod
    async def get_status(self, batch_api_job: BatchApiJob) -> BatchStatusInfo:
        """Get the current status of a batch job.

        Args:
            batch_api_job: The batch job to check

        Returns:
            Status info including request counts
        """
        pass

    @abstractmethod
    async def list_batches(self, limit: int = 20) -> list[BatchApiJob]:
        """List recent batch jobs.

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch jobs, most recent first
        """
        pass

    @abstractmethod
    async def get_results(self, batch_api_job: BatchApiJob) -> list[BatchResponse]:
        """Get results from a completed batch job.

        Args:
            batch_api_job: The batch job to get results from

        Returns:
            List of responses, including both successes and failures
        """
        pass

    @abstractmethod
    async def cancel(self, batch_api_job: BatchApiJob) -> bool:
        """Cancel a batch job.

        Args:
            batch_api_job: The batch job to cancel

        Returns:
            True if cancellation was successful, False otherwise
        """
        pass
