"""langasync - Async API integration library for multiple providers."""

from importlib.metadata import version

__version__ = version("langasync")

from langasync.settings import LangasyncSettings, langasync_settings
from langasync.core.batch_chain import BatchChainWrapper, batch_chain
from langasync.core.batch_poller import BatchPoller
from langasync.core.batch_service import BatchJobService
from langasync.core.batch_handle import BatchJobHandle, ProcessedResults
from langasync.core.batch_job_repository import FileSystemBatchJobRepository, BatchJobRepository
from langasync.providers.interface import BatchStatus

__all__ = [
    "batch_chain",
    "BatchPoller",
    "BatchJobService",
    "BatchStatus",
    "LangasyncSettings",
    "langasync_settings",
    "ProcessedResults",
]
