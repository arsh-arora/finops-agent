"""
Custom exceptions for the Memory System.
"""


class MemoryServiceError(Exception):
    """Base exception for memory service operations."""
    pass


class MemoryConfigurationError(MemoryServiceError):
    """Exception raised when memory service configuration is invalid."""
    pass


class MemoryNotFoundError(MemoryServiceError):
    """Exception raised when requested memory is not found."""
    pass


class MemoryStorageError(MemoryServiceError):
    """Exception raised when memory storage fails."""
    pass


class MemoryRetrievalError(MemoryServiceError):
    """Exception raised when memory retrieval fails."""
    pass


class GraphMemoryError(MemoryServiceError):
    """Exception raised when graph memory operations fail."""
    pass