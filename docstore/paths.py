"""Dot paths, index-clause syntax, and document traversal.

Index clause syntax
-------------------
``brand``             plain field
``specs[].color``     fan out over the array at ``specs``
``-specs[].price``    descending component

A ``[]`` marks where a multikey index fans out over an array.  Only one array
level per clause is supported; nested fan-out is rejected at index-creation
time.  Whether *data* actually contains an array is still detected per document
(an undeclared array forces the index multikey lazily, and a second array in a
compound index is rejected).

Query field paths never contain ``[]``.  Array element semantics in queries are
expressed with ``$elemMatch`` (same element) or dotted sibling conditions /
``$some`` (independent elements), never by punctuation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .errors import PathError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPONENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\])*)$")


def is_ident(text: str) -> bool:
    return bool(_IDENT.match(text))


# --------------------------------------------------------------------------
# Index clause
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PathStep:
    name: str
    fanout: bool = False


@dataclass(frozen=True)
class Clause:
    """One field of an index, e.g. ``specs[].color``."""

    steps: Tuple[PathStep, ...]
    descending: bool = False
    raw: str = ""

    @property
    def field(self) -> str:
        return ".".join(s.name for s in self.steps)

    @property
    def multikey_declared(self) -> bool:
        return any(s.fanout for s in self.steps)

    @property
    def fanout_paths(self) -> Tuple[str, ...]:
        out: List[str] = []
        prefix: List[str] = []
        for s in self.steps:
            prefix.append(s.name)
            if s.fanout:
                out.append(".".join(prefix))
        return tuple(out)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "multikey": self.multikey_declared,
            "direction": -1 if self.descending else 1,
        }


def parse_clause(text: str) -> Clause:
    if not isinstance(text, str) or not text.strip():
        raise PathError(f"index field must be a non-empty string, got {text!r}")
    raw = text.strip()
    descending = False
    if raw.startswith("-"):
        descending = True
        raw = raw[1:].strip()
    elif raw.startswith("+"):
        raw = raw[1:].strip()
    if not raw:
        raise PathError(f"index field {text!r} has no field name")
    if raw.startswith("$"):
        raise PathError(f"index field may not start with '$': {text!r}")
    steps: List[PathStep] = []
    for part in raw.split("."):
        m = _COMPONENT.match(part)
        if not m:
            raise PathError(
                f"invalid component {part!r} in index field {text!r}; components "
                f"must be identifiers, optionally followed by []")
        name, brackets = m.group(1), m.group(2)
        n_fan = len(brackets) // 2
        if n_fan > 1:
            raise PathError(
                f"index field {text!r} has more than one [] on component {name!r}; "
                f"declare a single array level per component")
        steps.append(PathStep(name=name, fanout=n_fan == 1))
    return Clause(steps=tuple(steps), descending=descending, raw=text.strip())


# --------------------------------------------------------------------------
# Query field path (no [] allowed)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryPath:
    raw: str
    steps: Tuple[str, ...]

    @property
    def dotted(self) -> str:
        return ".".join(self.steps)

    @property
    def root(self) -> str:
        return self.steps[0]

    def tail(self) -> "QueryPath":
        return QueryPath(raw=".".join(self.steps[1:]), steps=self.steps[1:])


def parse_query_path(raw: Any) -> QueryPath:
    if not isinstance(raw, str) or not raw.strip():
        raise PathError(f"query field must be a non-empty string, got {raw!r}")
    text = raw.strip()
    if text.startswith("$"):
        raise PathError(f"query field may not start with '$': {raw!r}")
    if "[" in text or "]" in text:
        raise PathError(
            f"query field {raw!r} contains [] -- array element semantics are "
            f"expressed with $elemMatch (same element) or dotted sibling "
            f"conditions / $some (independent elements), not with punctuation")
    steps = text.split(".")
    for step in steps:
        if not _IDENT.match(step):
            raise PathError(f"invalid path component {step!r} in {raw!r}")
    return QueryPath(raw=text, steps=tuple(steps))


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------

def get_path(doc: Any, dotted: str) -> List[Any]:
    """Every value reachable at ``dotted``, traversing arrays implicitly.

    Arrays met *before* the final component are traversed; an array at the
    final component is returned whole (callers flatten when they need
    elements).  A missing path returns ``[]``.
    """
    if not dotted:
        return [doc]
    return _collect(doc, dotted.split("."), 0)


def _collect(node: Any, parts: Sequence[str], i: int) -> List[Any]:
    if i == len(parts):
        return [node]
    if isinstance(node, list):
        out: List[Any] = []
        for el in node:
            out.extend(_collect(el, parts, i))
        return out
    if not isinstance(node, dict):
        return []
    part = parts[i]
    if part not in node:
        return []
    return _collect(node[part], parts, i + 1)


def get_path_one(doc: Any, dotted: str) -> Tuple[bool, Any]:
    if not dotted:
        return True, doc
    node = doc
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def flatten(values: Sequence[Any]) -> List[Any]:
    """Expand one level of arrays."""
    out: List[Any] = []
    for v in values:
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out


def path_exists(doc: Any, dotted: str) -> bool:
    return bool(get_path(doc, dotted))


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def set_path(doc: Any, dotted: str, value: Any) -> Tuple[bool, str]:
    """Set an unambiguous path.  Intermediate nodes must be objects.

    Returns (changed, reason).  Crossing an array is refused; use
    ``set_element`` for positional array updates.
    """
    if not dotted:
        return False, "cannot set the document root"
    parts = dotted.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False, f"path {dotted!r} does not exist"
        node = node[part]
        if isinstance(node, list):
            return False, (
                f"path {dotted!r} crosses an array at {part!r}; use a positional "
                f"update ('specs[2].stock') instead")
    if not isinstance(node, dict):
        return False, f"parent of {dotted!r} is not an object"
    leaf = parts[-1]
    if leaf in node and data_equal(node[leaf], value):
        return False, "no-op: value unchanged"
    node[leaf] = value
    return True, "set"


def unset_path(doc: Any, dotted: str) -> Tuple[bool, str]:
    parts = dotted.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False, f"path {dotted!r} does not exist"
        node = node[part]
        if isinstance(node, list):
            return False, f"path {dotted!r} crosses an array at {part!r}"
    if not isinstance(node, dict) or parts[-1] not in node:
        return False, f"path {dotted!r} does not exist"
    del node[parts[-1]]
    return True, "unset"


# --------------------------------------------------------------------------
# Positional update path: specs[2].stock
# --------------------------------------------------------------------------

_POSITIONAL = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")


@dataclass
class PositionalPath:
    """``specs[2].stock`` -> segments [("specs",2), ("stock",None)]."""

    raw: str
    segments: List[Tuple[str, Optional[int]]]

    @property
    def is_positional(self) -> bool:
        return any(idx is not None for _, idx in self.segments)

    def array_indices(self) -> List[Tuple[str, int]]:
        return [(name, idx) for name, idx in self.segments if idx is not None]


def parse_positional_path(raw: str) -> PositionalPath:
    if not isinstance(raw, str) or not raw.strip():
        raise PathError(f"update path must be a non-empty string, got {raw!r}")
    text = raw.strip()
    segs: List[Tuple[str, Optional[int]]] = []
    for part in text.split("."):
        m = _POSITIONAL.match(part)
        if m:
            segs.append((m.group(1), int(m.group(2))))
            continue
        if not _IDENT.match(part):
            raise PathError(
                f"invalid update path component {part!r}; use field names or "
                f"field[<index>] for positional array updates")
        segs.append((part, None))
    return PositionalPath(raw=text, segments=segs)


def apply_positional(doc: Any, ppath: PositionalPath, op: str,
                     value: Any = None) -> Tuple[bool, str]:
    """Apply set/unset along a positional path.

    Behaviour, documented and enforced:

    * ``specs[2].stock`` addresses index 2 of the *current* array.  A negative
      index counts from the end (``specs[-1]`` is not allowed -- use the last
      element via ``$push``/``$pop`` or read it first).
    * An out-of-range index is an error, not a silent append.
    * ``specs[2]`` (the element itself) with ``set`` replaces the whole element.
    * If an intermediate node is not an array where an index is given, error.
    * An index into a document (object) is an error.
    """
    node = doc
    for i, (name, idx) in enumerate(ppath.segments):
        last = i == len(ppath.segments) - 1
        if idx is None:
            if not isinstance(node, dict):
                return False, (
                    f"cannot address {name!r}: parent at {'.'.join(n for n,_ in ppath.segments[:i]) or '<root>'} "
                    f"is not an object")
            if last:
                if op == "unset":
                    if name not in node:
                        return False, f"path {ppath.raw!r} does not exist"
                    del node[name]
                    return True, "unset"
                if name in node and data_equal(node[name], value):
                    return False, "no-op: value unchanged"
                node[name] = value
                return True, "set"
            if name not in node:
                return False, f"path {ppath.raw!r} does not exist at {name!r}"
            node = node[name]
            continue
        # positional
        if not isinstance(node, dict) or name not in node:
            return False, f"path {ppath.raw!r} does not exist at {name!r}"
        arr = node[name]
        if not isinstance(arr, list):
            return False, (
                f"{name!r} is not an array (it is {type(arr).__name__}); "
                f"positional update is only valid on arrays")
        if idx < 0 or idx >= len(arr):
            return False, (
                f"index {idx} out of range for {name!r} (length {len(arr)}); "
                f"out-of-range positional updates are rejected, never appended")
        if last:
            if op == "unset":
                arr.pop(idx)
                return True, "unset-element"
            if data_equal(arr[idx], value):
                return False, "no-op: value unchanged"
            arr[idx] = value
            return True, "set-element"
        node = arr[idx]
    return False, "empty update path"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def data_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(data_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(data_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def validate_update_path(dotted: str) -> None:
    if not dotted or dotted.startswith("$"):
        raise PathError(f"invalid update path {dotted!r}")
    parse_positional_path(dotted)
