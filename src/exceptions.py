class MdHybridSearchError(Exception):
    """Base exception for md-hybrid-search."""
    pass

class ConfigMismatchError(MdHybridSearchError):
    """Raised when the collection configuration diverges from the stored metadata."""
    pass

class IndexCorruptionError(MdHybridSearchError):
    """Raised when the index (SQLite or ChromaDB) is corrupted or has an incompatible schema."""
    pass

class EmbeddingError(MdHybridSearchError):
    """Raised when an error occurs during embedding generation."""
    pass

class SourceNotFoundError(MdHybridSearchError, FileNotFoundError):
    """Raised when a specified source directory is not found."""
    pass
