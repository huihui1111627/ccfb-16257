"""docstore -- a document store with multikey indexes over nested arrays."""

from .collection import Collection
from .errors import (
    DocStoreError,
    DuplicateDocumentError,
    DuplicateKeyError,
    IndexError_,
    MultikeyUniqueError,
    ParallelArraysError,
    PathError,
    QueryError,
    StorageCorruptionError,
)
from .index import Extracted, Index, IndexDef, extract_keys
from .paths import Clause, parse_clause, parse_positional_path, parse_query_path
from .query import matches, parse_query
from .storage import Storage

__all__ = [
    "Collection", "DocStoreError", "DuplicateDocumentError", "DuplicateKeyError",
    "IndexError_", "MultikeyUniqueError", "ParallelArraysError", "PathError",
    "QueryError", "StorageCorruptionError", "Extracted", "Index", "IndexDef",
    "extract_keys", "Clause", "parse_clause", "parse_positional_path",
    "parse_query_path", "matches", "parse_query", "Storage",
]
