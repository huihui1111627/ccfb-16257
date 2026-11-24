"""Query language: parsing, validation and the reference matcher.

Two array predicates, deliberately distinct
-------------------------------------------
``$elemMatch`` -- **same element**::

    {"specs": {"$elemMatch": {"color": "red", "size": "XL", "stock": {"$gt": 0}}}}

    one element of ``specs`` is red and XL and has stock.  This is what the ops
    team meant.

dotted sibling conditions / ``$some`` -- **independent elements**::

    {"specs.color": "red", "specs.size": "XL"}
    {"$some": {"specs.color": "red", "specs.size": "XL"}}

    some element is red, and (possibly another) element is XL.  This is the
    production query that returned "red S plus a different colour XL".  It is
    spelled ``$some`` when the author wants to be explicit that different
    elements may satisfy the different conditions.

``$every`` -- every element satisfies all conditions::

    {"$every": {"specs.size": {"$ne": "XXS"}}}

    vacuously true for an empty array and for a missing path (documented).

The plan output labels each array predicate with which semantics it used, so
the difference is visible without reading the matcher.

Matcher guarantees
------------------
One reference implementation decides truth.  The index layer only chooses
*candidate* documents; every candidate is re-checked here.  Therefore the
"indexed" and "full scan" paths cannot disagree by construction, and the
conformance harness can compare them document by document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import values as V
from .errors import QueryError
from .paths import QueryPath, flatten, get_path, parse_query_path

CMP_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
EXTRA_FIELD_OPS = {"$exists", "$type", "$size", "$all", "$elemMatch", "$not", "$regex"}
FIELD_OPS = CMP_OPS | EXTRA_FIELD_OPS
LOGICAL_OPS = {"$and", "$or", "$nor"}
SCOPE_OPS = {"$some", "$every"}
TOP_OPS = LOGICAL_OPS | SCOPE_OPS

#: operators that test ONE element/whole value and must agree on the same one
ELEMENT_OPS = {"$eq", "$gt", "$gte", "$lt", "$lte", "$in", "$regex"}
#: operators that test the field as a set / need whole arrays
SET_OPS = {"$ne", "$nin", "$elemMatch", "$all", "$size", "$not", "$exists", "$type"}


# --------------------------------------------------------------------------
# Parsed tree
# --------------------------------------------------------------------------

@dataclass
class FieldCond:
    path: QueryPath
    ops: Dict[str, Any]

    @property
    def field(self) -> str:
        return self.path.dotted


@dataclass
class And:
    children: List[Any]


@dataclass
class Or:
    children: List[Any]


@dataclass
class Nor:
    children: List[Any]


@dataclass
class Some:
    """Independent witnesses: each condition may bind a different element."""

    conditions: List[FieldCond]
    array_path: Optional[str] = None
    explicit: bool = False


@dataclass
class Every:
    conditions: List[FieldCond]
    array_path: Optional[str] = None


Node = Any


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _is_op_object(value: Any) -> bool:
    return isinstance(value, dict) and any(
        isinstance(k, str) and k.startswith("$") for k in value)


def _validate_ops(path: QueryPath, ops: Dict[str, Any]) -> None:
    if not ops:
        raise QueryError(f"empty condition on field {path.raw!r}")
    for op, arg in ops.items():
        if not (isinstance(op, str) and op.startswith("$")):
            raise QueryError(
                f"field {path.raw!r} mixes a bare value with operators; write "
                f"{{'$eq': ...}} instead")
        if op not in FIELD_OPS:
            raise QueryError(
                f"unknown operator {op!r} on field {path.raw!r}; known: "
                f"{sorted(FIELD_OPS)}")
        if op in ("$gt", "$gte", "$lt", "$lte"):
            if isinstance(arg, (dict, list)) or arg is None or isinstance(arg, bool):
                raise QueryError(
                    f"{op} on {path.raw!r} needs a number, string or bool; got "
                    f"{type(arg).__name__}")
        if op in ("$in", "$nin"):
            if not isinstance(arg, list):
                raise QueryError(f"{op} on {path.raw!r} needs an array")
        if op == "$all":
            if not isinstance(arg, list) or not arg:
                raise QueryError(f"$all on {path.raw!r} needs a non-empty array")
        if op == "$exists":
            if not isinstance(arg, bool):
                raise QueryError(f"$exists on {path.raw!r} needs true or false")
        if op == "$type":
            names = arg if isinstance(arg, list) else [arg]
            valid = set(V.TYPE_NAMES.values())
            for n in names:
                if n not in valid:
                    raise QueryError(
                        f"$type {n!r} on {path.raw!r} is invalid; valid: "
                        f"{sorted(valid)}")
        if op == "$size":
            if isinstance(arg, bool) or not isinstance(arg, int) or arg < 0:
                raise QueryError(f"$size on {path.raw!r} needs a non-negative int")
        if op == "$regex":
            if not isinstance(arg, str):
                raise QueryError(f"$regex on {path.raw!r} needs a string pattern")
            try:
                re.compile(arg)
            except re.error as exc:
                raise QueryError(f"bad $regex {arg!r} on {path.raw!r}: {exc}") from exc
        if op == "$elemMatch":
            if not isinstance(arg, dict) or not arg:
                raise QueryError(
                    f"$elemMatch on {path.raw!r} needs a non-empty object of "
                    f"element conditions")
            _validate_elem_body(path, arg)
        if op == "$not":
            if not isinstance(arg, dict) or not arg:
                raise QueryError(
                    f"$not on {path.raw!r} needs a non-empty operator object")
            if any(not k.startswith("$") for k in arg):
                raise QueryError(f"$not on {path.raw!r} must contain only operators")
            _validate_ops(path, arg)


def _validate_elem_body(path: QueryPath, body: Dict[str, Any]) -> None:
    for k, v in body.items():
        if isinstance(k, str) and k.startswith("$"):
            # element itself is compared with an operator, e.g. {"$gt": 5}
            _validate_ops(path, {k: v})
            continue
        sub = parse_query_path(k)
        ops = v if _is_op_object(v) else {"$eq": v}
        _validate_ops(sub, ops)


def _make_field_cond(key: str, raw: Any) -> FieldCond:
    path = parse_query_path(key)
    ops = raw if _is_op_object(raw) else {"$eq": raw}
    _validate_ops(path, ops)
    return FieldCond(path=path, ops=dict(ops))


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def parse_query(query: Any) -> Node:
    if not isinstance(query, dict):
        raise QueryError("query must be a JSON object")
    if not query:
        return And([])
    parts: List[Node] = []
    for key, raw in query.items():
        if isinstance(key, str) and key.startswith("$"):
            if key in LOGICAL_OPS:
                if not isinstance(raw, list) or not raw:
                    raise QueryError(f"{key} needs a non-empty array of subqueries")
                children = [parse_query(q) for q in raw]
                parts.append({"$and": And, "$or": Or, "$nor": Nor}[key](children))
                continue
            if key in SCOPE_OPS:
                parts.append(_parse_scope(key, raw))
                continue
            if key in FIELD_OPS:
                raise QueryError(
                    f"top-level {key!r} is ambiguous: it needs a field. Write "
                    f"{{'field': {{'{key}': ...}}}}. To test any element of an array, "
                    f"use $elemMatch (same element) or dotted conditions / $some "
                    f"(independent elements).")
            raise QueryError(f"unknown top-level operator {key!r}")
        parts.append(_make_field_cond(key, raw))
    return parts[0] if len(parts) == 1 else And(parts)


def _parse_scope(op: str, raw: Any) -> Node:
    if not isinstance(raw, dict) or not raw:
        raise QueryError(
            f"{op} needs a non-empty object of field conditions, e.g. "
            f"{{'{op}': {{'specs.color': 'red', 'specs.size': 'XL'}}}}")
    conds: List[FieldCond] = []
    for k, v in raw.items():
        if isinstance(k, str) and k.startswith("$"):
            raise QueryError(
                f"inside {op}, nested logical operator {k!r} is not supported; "
                f"use $and/$or around separate {op} clauses")
        conds.append(_make_field_cond(k, v))
    arrays = {c.path.root for c in conds}
    if len(arrays) > 1:
        raise QueryError(
            f"{op} conditions must all refer to the same array; got roots "
            f"{sorted(arrays)}")
    if op == "$some":
        return Some(conds, array_path=conds[0].path.root, explicit=True)
    return Every(conds, array_path=conds[0].path.root)


# --------------------------------------------------------------------------
# Value-level matching
# --------------------------------------------------------------------------

def _cmp(value: Any, op: str, arg: Any) -> bool:
    a, b = V.encode_key(value), V.encode_key(arg)
    if op == "$eq":
        return a == b
    if op == "$ne":
        return a != b
    if op == "$gt":
        return a > b
    if op == "$gte":
        return a >= b
    if op == "$lt":
        return a < b
    return a <= b  # $lte


def _single(value: Any, op: str, arg: Any) -> bool:
    """Apply one operator to one already-extracted value."""
    if op in CMP_OPS:
        return _cmp(value, op, arg)
    if op == "$type":
        names = arg if isinstance(arg, list) else [arg]
        return V.value_type(value) in names
    if op == "$regex":
        return isinstance(value, str) and re.search(arg, value) is not None
    if op == "$size":
        return isinstance(value, list) and len(value) == arg
    raise AssertionError(op)


def _values_for(doc: Any, path: QueryPath) -> Tuple[bool, List[Any], List[Any]]:
    """Return (exists, whole_values, flattened_values)."""
    matched = get_path(doc, path.dotted)
    if not matched:
        return False, [], []
    return True, matched, flatten(matched)


def _match_ops(doc: Any, path: QueryPath, ops: Dict[str, Any]) -> bool:
    """Evaluate a set of operators on a path of one document.

    Element semantics within *one* condition object (documented):

    * The *element* operators (``$eq``/``$gt``/``$gte``/``$lt``/``$lte``/
      ``$in``/``$regex``) must share one element.  ``{"price": {"$gt": 10,
      "$lt": 20}}`` means one element in (10, 20); for a document that is a
      single scalar this is the obvious reading, and for an array it is the
      reading that lets an index intersect the two bounds safely.
    * Set operators (``$ne``, ``$nin``, ``$size``, ``$all``, ``$elemMatch``,
      ``$not``) are independent of that element and of each other.
    * On a document whose value is *not* an array, all comparisons apply to
      the one value, so behaviour is the familiar one.

    To say "some element > 10 and some (possibly different) element < 20",
    write two conditions (``$and`` of two field conditions), not one.
    """
    exists, whole, flat = _values_for(doc, path)
    element_ops = {k: v for k, v in ops.items() if k in ELEMENT_OPS}
    set_ops = {k: v for k, v in ops.items() if k in SET_OPS}

    # --- set-level operators (each independent) -------------------------
    for op, arg in set_ops.items():
        if op == "$exists":
            if exists is not arg:
                return False
            continue
        if op == "$ne":
            if exists and any(_cmp(v, "$eq", arg) for v in whole + flat):
                return False
            continue
        if op == "$nin":
            if exists and any(_cmp(v, "$eq", a) for v in whole + flat for a in arg):
                return False
            continue
        if op == "$not":
            if _match_ops(doc, path, arg):
                return False
            continue
        if op == "$elemMatch":
            if not exists or not _elem_match(whole, arg):
                return False
            continue
        if op == "$size":
            if not exists or not any(isinstance(v, list) and len(v) == arg
                                     for v in whole):
                return False
            continue
        if op == "$all":
            if not exists or not _all_match(whole, flat, arg):
                return False
            continue
        if op == "$type":
            if not exists or not any(_single(v, "$type", arg) for v in whole + flat):
                return False
            continue
        raise AssertionError(op)

    # --- element operators: one element must satisfy all ----------------
    if element_ops:
        # If the value is not an array, the "element" is the value itself.
        candidates: List[Any] = []
        if exists:
            candidates = flat
        matched_element = False
        for v in candidates:
            ok = True
            for op, arg in element_ops.items():
                if op == "$in":
                    if not any(_cmp(v, "$eq", a) for a in arg):
                        ok = False
                        break
                elif not _single(v, op, arg):
                    ok = False
                    break
            if ok:
                matched_element = True
                break
        if not matched_element:
            return False
    return True


def _all_match(whole: List[Any], flat: List[Any], args: List[Any]) -> bool:
    for v in whole:
        if isinstance(v, list) and all(any(_cmp(el, "$eq", a) for el in v) for a in args):
            return True
    return all(any(_cmp(el, "$eq", a) for el in flat) for a in args)


def _elem_match(whole: List[Any], body: Dict[str, Any]) -> bool:
    for v in whole:
        elements = v if isinstance(v, list) else [v]
        for el in elements:
            if _elem_body_matches(el, body):
                return True
    return False


def _elem_body_matches(el: Any, body: Dict[str, Any]) -> bool:
    for k, v in body.items():
        if isinstance(k, str) and k.startswith("$"):
            if not _match_ops(el, QueryPath(raw="", steps=()), {k: v}):
                return False
            continue
        sub = parse_query_path(k)
        ops = v if _is_op_object(v) else {"$eq": v}
        if not _match_ops(el, sub, ops):
            return False
    return True


# --------------------------------------------------------------------------
# Tree evaluation
# --------------------------------------------------------------------------

def matches(doc: Any, node: Node) -> bool:
    if isinstance(node, And):
        return all(matches(doc, c) for c in node.children)
    if isinstance(node, Or):
        return any(matches(doc, c) for c in node.children)
    if isinstance(node, Nor):
        return not any(matches(doc, c) for c in node.children)
    if isinstance(node, FieldCond):
        return _match_ops(doc, node.path, node.ops)
    if isinstance(node, Some):
        for c in node.conditions:
            if not _match_ops(doc, c.path, c.ops):
                return False
        return True
    if isinstance(node, Every):
        return _every(doc, node)
    raise AssertionError(type(node))


def _every(doc: Any, node: Every) -> bool:
    root = node.array_path
    if root is None:
        return True
    matched = get_path(doc, root)
    if not matched:
        return True  # vacuous: no array, no counterexample (documented)
    for v in matched:
        elements = v if isinstance(v, list) else [v]
        for el in elements:
            for c in node.conditions:
                rest = c.path.steps[1:]
                sub = QueryPath(raw=".".join(rest), steps=tuple(rest))
                if not _match_ops(el, sub, c.ops):
                    return False
    return True


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def walk_nodes(node: Node) -> Iterable[Node]:
    yield node
    if isinstance(node, (And, Or, Nor)):
        for c in node.children:
            yield from walk_nodes(c)
    elif isinstance(node, (Some, Every)):
        for c in node.conditions:
            yield c


def field_conds(node: Node) -> List[FieldCond]:
    return [n for n in walk_nodes(node) if isinstance(n, FieldCond)]


def elem_matches(node: Node) -> List[FieldCond]:
    out: List[FieldCond] = []
    for n in walk_nodes(node):
        if isinstance(n, FieldCond) and "$elemMatch" in n.ops:
            out.append(n)
    return out


def array_semantics(node: Node) -> List[dict]:
    """Describe every array predicate and which semantics it uses."""
    out: List[dict] = []
    for n in walk_nodes(node):
        if isinstance(n, FieldCond):
            if "$elemMatch" in n.ops:
                out.append({
                    "field": n.path.raw,
                    "semantics": "same-element",
                    "operator": "$elemMatch",
                    "conditions": [
                        {k: v} for k, v in n.ops["$elemMatch"].items()
                    ],
                })
            elif any(op in SET_OPS for op in n.ops):
                out.append({
                    "field": n.path.raw,
                    "semantics": "set-level",
                    "operator": sorted(op for op in n.ops if op in SET_OPS),
                })
        elif isinstance(n, Some):
            for c in n.conditions:
                out.append({
                    "field": c.path.raw,
                    "semantics": "independent-elements",
                    "operator": "$some" if n.explicit else "dotted",
                })
        elif isinstance(n, Every):
            for c in n.conditions:
                out.append({
                    "field": c.path.raw,
                    "semantics": "every-element",
                    "operator": "$every",
                })
    return out


def node_to_dict(node: Node) -> Any:
    if isinstance(node, And):
        return {"$and": [node_to_dict(c) for c in node.children]}
    if isinstance(node, Or):
        return {"$or": [node_to_dict(c) for c in node.children]}
    if isinstance(node, Nor):
        return {"$nor": [node_to_dict(c) for c in node.children]}
    if isinstance(node, FieldCond):
        return {node.path.raw: node.ops}
    if isinstance(node, Some):
        return {"$some": {c.path.raw: c.ops for c in node.conditions}}
    if isinstance(node, Every):
        return {"$every": {c.path.raw: c.ops for c in node.conditions}}
    return repr(node)


def explain_semantics(node: Node) -> str:
    if isinstance(node, And):
        return " AND ".join(explain_semantics(c) for c in node.children) or "TRUE"
    if isinstance(node, Or):
        return "(" + " OR ".join(explain_semantics(c) for c in node.children) + ")"
    if isinstance(node, Nor):
        return "NOT (" + " OR ".join(explain_semantics(c) for c in node.children) + ")"
    if isinstance(node, FieldCond):
        return f"{node.path.raw} {_ops_text(node.ops)}"
    if isinstance(node, Some):
        inner = " AND ".join(f"{c.path.raw} {_ops_text(c.ops)}"
                             for c in node.conditions)
        return (f"SOME elements of {node.array_path!r} independently satisfy "
                f"({inner})")
    if isinstance(node, Every):
        inner = " AND ".join(f"{c.path.raw} {_ops_text(c.ops)}"
                             for c in node.conditions)
        return f"EVERY element of {node.array_path!r} satisfies ({inner})"
    return repr(node)


def _ops_text(ops: Dict[str, Any]) -> str:
    parts = []
    for op, arg in ops.items():
        if op == "$elemMatch":
            inner = " AND ".join(f"{k}={v!r}" for k, v in arg.items())
            parts.append(f"$elemMatch({inner})")
        elif op == "$eq":
            parts.append(f"= {arg!r}")
        else:
            parts.append(f"{op} {arg!r}")
    return " ".join(parts)
