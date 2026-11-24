"""Exception types and error codes."""

from __future__ import annotations


class DocStoreError(Exception):
    code = "DOCSTORE_ERROR"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


class QueryError(DocStoreError):
    code = "QUERY_ERROR"


class PathError(DocStoreError):
    code = "PATH_ERROR"


class IndexError_(DocStoreError):
    code = "INDEX_ERROR"


class DuplicateKeyError(DocStoreError):
    code = "DUPLICATE_KEY"

    def __init__(self, index: str, key, existing_id, new_id):
        super().__init__(
            f"unique index {index!r} rejects duplicate key {key!r} "
            f"(existing doc {existing_id!r}, new doc {new_id!r})",
            index=index,
            key=key,
            existing_id=existing_id,
            new_id=new_id,
        )


class MultikeyUniqueError(DocStoreError):
    code = "MULTIKEY_UNIQUE_UNSUPPORTED"

    def __init__(self, index: str, doc_id, n_keys: int):
        super().__init__(
            f"unique index {index!r} cannot index doc {doc_id!r}: the indexed path is an "
            f"array and yields {n_keys} keys. A unique index on an array field is "
            f"rejected because one document would occupy many keys and the uniqueness "
            f"constraint would be meaningless. Use a scalar field, or a non-unique "
            f"multikey index.",
            index=index,
            doc_id=doc_id,
            keys=n_keys,
        )


class ParallelArraysError(DocStoreError):
    code = "PARALLEL_ARRAYS"

    def __init__(self, index: str, field_a: str, field_b: str):
        super().__init__(
            f"refusing to build compound index {index!r} on ({field_a}, {field_b}): both "
            f"fields are arrays. A compound multikey index would need the cross product of "
            f"the element sets, so one 10x10 pair of arrays produces 100 index entries per "
            f"document. Split into two single-field indexes and intersect them instead.",
            index=index,
            fields=[field_a, field_b],
        )


class DocumentTooLargeError(DocStoreError):
    code = "DOCUMENT_TOO_LARGE"


class StorageCorruptionError(DocStoreError):
    code = "STORAGE_CORRUPTION"


class DuplicateDocumentError(DocStoreError):
    code = "DUPLICATE_DOCUMENT"
