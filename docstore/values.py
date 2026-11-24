"""Canonical total ordering for JSON-ish document values.

Every value is mapped to a *sortable key*: a hashable, totally ordered tuple
``(type_rank, *payload)``.  The index, planner and query matcher all agree on
one encoding, which is what keeps "equal" meaning the same thing everywhere.

Documented type order (ascending)::

    MISSING < null < NaN < number < string < bool < object < array < MAX

Notes / deliberate decisions
-----------------------------
* ``bool`` is not a number: ``True`` never equals ``1``.
* Numbers compare numerically.  Floats go through their shortest round-trip
  decimal text, so a price stored as ``199.0`` is found by ``$eq: 199`` too.
  This is what a human writing a query expects; it is not IEEE bit equality.
* ``NaN`` gets its own slot *below* every real number and equals itself.  IEEE
  says all NaN comparisons are false, but an index key must be findable.
* ``±Infinity`` sorts normally.
* A missing path gets the sentinel rank ``MISSING``, never the value ``null``.
  ``$exists`` can tell them apart; equality with ``null`` also matches missing
  (documented, MongoDB-compatible) but the *plan* distinguishes them.
* An empty array is a real value: ``(RANK_ARRAY, ())``, sorts before ``[1]``.
  This lets ``$eq: []`` and ``$exists: true`` stay answerable.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

RANK_MINKEY = -2
RANK_MISSING = 0
RANK_NULL = 1
RANK_NUMBER = 2
RANK_STRING = 3
RANK_BOOL = 4
RANK_OBJECT = 5
RANK_ARRAY = 6
RANK_MAXKEY = 99

TYPE_NAMES = {
    RANK_MINKEY: "minkey",
    RANK_MISSING: "missing",
    RANK_NULL: "null",
    RANK_NUMBER: "number",
    RANK_STRING: "string",
    RANK_BOOL: "bool",
    RANK_OBJECT: "object",
    RANK_ARRAY: "array",
    RANK_MAXKEY: "maxkey",
}

MINKEY = (RANK_MINKEY,)
MAXKEY = (RANK_MAXKEY,)
MISSING_KEY = (RANK_MISSING, "missing")
EMPTY_ARRAY_KEY = (RANK_ARRAY, ())


class UnorderableValue(TypeError):
    """Raised when a value cannot be a document/index value."""


def rank_of(value: Any) -> int:
    if value is None:
        return RANK_NULL
    if isinstance(value, bool):
        return RANK_BOOL
    if isinstance(value, (int, float, Decimal)):
        return RANK_NUMBER
    if isinstance(value, str):
        return RANK_STRING
    if isinstance(value, dict):
        return RANK_OBJECT
    if isinstance(value, (list, tuple)):
        return RANK_ARRAY
    raise UnorderableValue(f"cannot order/hash value of type {type(value).__name__}")


def normalize_number(value: Any) -> Decimal | None:
    """Decimal for a numeric value; ``None`` means NaN."""
    if isinstance(value, bool):  # pragma: no cover - callers guard
        raise UnorderableValue("bool is not a number here")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        return None if value.is_nan() else value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return Decimal("Infinity") if value > 0 else Decimal("-Infinity")
        return Decimal(str(value))
    raise UnorderableValue(f"not a number: {type(value).__name__}")


def encode_key(value: Any) -> tuple:
    rank = rank_of(value)
    if rank == RANK_NULL:
        return (RANK_NULL,)
    if rank == RANK_NUMBER:
        dec = normalize_number(value)
        if dec is None:
            return (RANK_NUMBER, 0)          # NaN slot, below (RANK_NUMBER, 1, ...)
        return (RANK_NUMBER, 1, dec)
    if rank == RANK_STRING:
        return (RANK_STRING, value)
    if rank == RANK_BOOL:
        return (RANK_BOOL, 1 if value else 0)
    if rank == RANK_OBJECT:
        return (RANK_OBJECT, tuple((k, encode_key(v)) for k, v in sorted(value.items())))
    if rank == RANK_ARRAY:
        return (RANK_ARRAY, tuple(encode_key(v) for v in value))
    raise AssertionError("unreachable")


def is_nan_key(key: tuple) -> bool:
    return key[0] == RANK_NUMBER and len(key) == 2


def key_type_name(key: tuple) -> str:
    return TYPE_NAMES.get(key[0], f"unknown({key[0]})")


def is_min_key(key: tuple) -> bool:
    return key == MINKEY


def is_max_key(key: tuple) -> bool:
    return key == MAXKEY


def compare(a: Any, b: Any) -> int:
    ka, kb = encode_key(a), encode_key(b)
    return (ka > kb) - (ka < kb)


def same_value(a: Any, b: Any) -> bool:
    """Type-bracketed equality: ``1 != "1"``, ``True != 1``."""
    return encode_key(a) == encode_key(b)


def value_type(value: Any) -> str:
    return TYPE_NAMES[rank_of(value)]


_JSON_SCALARS = (str, int, float, bool, type(None))


def ensure_document_value(value: Any, where: str = "value") -> None:
    if isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise UnorderableValue(f"{where}: non-finite float {value!r} is not storable")
        return
    if isinstance(value, Decimal):
        raise UnorderableValue(
            f"{where}: Decimal is not a storable document type; use int/float/str")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise UnorderableValue(f"{where}: object keys must be strings, got {k!r}")
            ensure_document_value(v, f"{where}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            ensure_document_value(v, f"{where}[{i}]")
        return
    raise UnorderableValue(f"{where}: unsupported value type {type(value).__name__}")


# ---- rendering -----------------------------------------------------------

def describe_key(key: tuple) -> Any:
    """Human-readable form of an encoded key."""
    if not isinstance(key, tuple):
        return key
    rank = key[0]
    if rank == RANK_MINKEY:
        return "<min>"
    if rank == RANK_MAXKEY:
        return "<max>"
    if rank == RANK_MISSING:
        return "<missing>"
    if rank == RANK_NULL:
        return None
    if rank == RANK_NUMBER:
        if is_nan_key(key):
            return "NaN"
        dec = key[2]
        if dec.is_finite() and dec == dec.to_integral_value():
            return int(dec)
        return str(dec)
    if rank == RANK_STRING:
        return key[1]
    if rank == RANK_BOOL:
        return bool(key[1])
    if rank == RANK_OBJECT:
        return {k: describe_key(v) for k, v in key[1]}
    if rank == RANK_ARRAY:
        return [describe_key(v) for v in key[1]]
    return repr(key)


def describe_value(value: Any) -> Any:
    try:
        return describe_key(encode_key(value))
    except UnorderableValue:
        return f"<{type(value).__name__}>"


def key_to_json(key: Any) -> Any:
    """Lossless JSON encoding of a key tuple (for manifests / UI)."""
    if isinstance(key, tuple):
        return {"t": [key_to_json(x) for x in key]}
    if isinstance(key, Decimal):
        return {"d": str(key)}
    if isinstance(key, bool):
        return {"b": key}
    if isinstance(key, int):
        return {"i": key}
    if key is None:
        return None
    if isinstance(key, str):
        return {"s": key}
    raise AssertionError(f"unexpected key part {key!r}")


def key_from_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [key_from_json(x) for x in obj]
    if "t" in obj:
        return tuple(key_from_json(x) for x in obj["t"])
    if "d" in obj:
        return Decimal(obj["d"])
    if "b" in obj:
        return bool(obj["b"])
    if "i" in obj:
        return int(obj["i"])
    if "s" in obj:
        return str(obj["s"])
    raise AssertionError(f"unexpected JSON key part {obj!r}")


def min_value(values: Iterable[Any]) -> Any:
    return min(values, key=encode_key)


def max_value(values: Iterable[Any]) -> Any:
    return max(values, key=encode_key)


def try_decimal(text: Any) -> Decimal | None:
    if not isinstance(text, str):
        return None
    try:
        d = Decimal(text.strip())
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None
