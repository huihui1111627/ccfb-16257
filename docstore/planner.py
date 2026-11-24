"""Query planning and execution.

The planner's job is to pick *candidate documents* cheaply.  It is never
allowed to decide truth: every candidate is re-checked by
``query.matches``.  So a plan may over-scan, never under-scan.

Multikey bound merging -- the rule
----------------------------------
With a multikey index a document owns one key *per array element*.  That
changes when two predicates may be combined into one tight key range.

**Safe to merge** (intersect the intervals, one scan):

* the index is not multikey (one key per document, so all predicates test the
  same scalar);
* the predicates come from one ``$elemMatch`` on the indexed array, i.e. the
  same element must satisfy all of them;
* different *leading* equality clauses of an aligned compound index under
  ``$elemMatch``.

**Unsafe to merge** (would lose documents):

* two range predicates on the same multikey field written as independent
  witnesses, e.g. ``{"specs.price": {"$gt": 10, "$lt": 20}}``.  The matcher
  requires *some* element > 10 and *some* element < 20, possibly different
  elements.  A document ``[{"price": 1}, {"price": 30}]`` matches, but no
  element lies in (10, 20), so scanning the merged interval (10, 20) would
  miss it.  The planner therefore scans each predicate's interval separately
  and *unions* the results, leaving the final AND to the matcher.
* two clauses of a multikey compound index under independent semantics, e.g.
  ``$some {"specs.color": "red", "specs.stock": {"$gt": 0}}``.  A document
  with elements ``(red, 0)`` and ``(blue, 5)`` matches, but the aligned entry
  ``(red, >0)`` does not exist.  Only the leading clause shapes the scan; the
  rest are residual filters.

In both unsafe cases the scan is a strict superset and the matcher trims it:
more rows read, never a missing row.  The plan states explicitly which rule
applied ("merged" vs "union" vs "residual").

Index intersection is used when no single index covers several predicates:
scan each index separately and intersect the doc-id sets.  This is the
supported alternative to a forbidden parallel-array compound index.

Sorting
-------
An index can supply the order when the sort fields are a prefix of its
clauses and all directions agree (forward) or all are opposite (reverse).
For a multikey index the document order is by its smallest indexed element
(ascending) or largest (descending); each document appears once.  In-memory
sorting uses the same rule, so both paths agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import values as V
from .errors import QueryError
from .index import Index, IndexDef, MISSING
from .paths import QueryPath
from .query import (
    And, Every, FieldCond, Nor, Or, Some,
    _is_op_object, field_conds, matches, parse_query_path, walk_nodes,
)

Key = Tuple[tuple, ...]


# --------------------------------------------------------------------------
# Predicates gathered from the parsed tree
# --------------------------------------------------------------------------

@dataclass
class Pred:
    field: str
    ops: Dict[str, Any]
    scope: str                 # "same-element" | "independent"
    align_root: Optional[str]
    source: str
    usable: bool = True

    @property
    def element_ops(self) -> Dict[str, Any]:
        return {k: v for k, v in self.ops.items() if k in _ELEMENT_OPS}

    @property
    def set_ops(self) -> Dict[str, Any]:
        return {k: v for k, v in self.ops.items() if k in _SET_OPS}


_ELEMENT_OPS = {"$eq", "$gt", "$gte", "$lt", "$lte", "$in", "$regex"}
_SET_OPS = {"$ne", "$nin", "$elemMatch", "$all", "$size", "$not", "$exists",
            "$type"}


def gather_predicates(tree: Any) -> Tuple[List[Pred], List[str]]:
    """Collect conjunctive predicates usable for index bounds.

    Anything under $or/$nor, and any $every, is *not* usable for narrowing;
    it still runs in the matcher.  Returns (predicates, notes).
    """
    preds: List[Pred] = []
    notes: List[str] = []

    def rec(node: Any, conjunctive: bool) -> None:
        if isinstance(node, And):
            for c in node.children:
                rec(c, conjunctive)
            return
        if isinstance(node, (Or, Nor)):
            notes.append(f"{type(node).__name__.lower()} subtree not usable for "
                         f"index bounds")
            return
        if isinstance(node, Every):
            notes.append("$every is not usable for index bounds (it constrains "
                         "every element, not some element)")
            return
        if isinstance(node, Some):
            for c in node.conditions:
                preds.append(Pred(c.path.dotted, dict(c.ops), "independent",
                                  node.array_path, "$some"))
            return
        if isinstance(node, FieldCond):
            if "$elemMatch" in node.ops:
                body = node.ops["$elemMatch"]
                root = node.path.dotted
                for k, v in body.items():
                    if isinstance(k, str) and k.startswith("$"):
                        notes.append(
                            f"$elemMatch operator predicate {k!r} on the element "
                            f"itself is residual")
                        continue
                    ops = v if _is_op_object(v) else {"$eq": v}
                    preds.append(Pred(f"{root}.{k}", dict(ops), "same-element",
                                      root, "$elemMatch"))
                # other ops alongside $elemMatch (rare) are residual
                for op in node.ops:
                    if op != "$elemMatch":
                        notes.append(f"operator {op} next to $elemMatch is residual")
                return
            preds.append(Pred(node.path.dotted, dict(node.ops), "independent",
                              None, "plain"))
            return
        notes.append(f"unsupported node {type(node).__name__} is residual")

    rec(tree, True)
    return preds, notes


# --------------------------------------------------------------------------
# Bound construction
# --------------------------------------------------------------------------

@dataclass
class KeyRange:
    lo: Optional[Key]
    hi: Optional[Key]
    lo_inclusive: bool
    hi_inclusive: bool

    def describe(self, n: int) -> str:
        lo = "-inf" if self.lo is None else _fmt(self.lo, n)
        hi = "+inf" if self.hi is None else _fmt(self.hi, n)
        lb = "[" if self.lo_inclusive else "("
        hb = "]" if self.hi_inclusive else ")"
        return f"{lb}{lo} .. {hi}{hb}"


def _fmt(key: Key, n: int) -> str:
    parts = []
    for c in key:
        if c == V.MINKEY:
            parts.append("-inf")
        elif c == V.MAXKEY:
            parts.append("+inf")
        else:
            parts.append(repr(V.describe_key(c)))
    return "[" + ", ".join(parts) + "]"


def _enc(value: Any) -> tuple:
    return V.encode_key(value)


def _eq_points(ops: Dict[str, Any]) -> Optional[List[Any]]:
    if "$eq" in ops:
        return [ops["$eq"]]
    if "$in" in ops:
        return list(ops["$in"])
    return None


def _range_interval(ops: Dict[str, Any]) -> Optional[Tuple[Any, Any, bool, bool]]:
    lo = hi = None
    lo_inc = hi_inc = True
    has = False
    if "$gte" in ops:
        lo, lo_inc, has = ops["$gte"], True, True
    if "$gt" in ops:
        val = ops["$gt"]
        if lo is None or _cmpv(val, lo) > 0:
            lo, lo_inc = val, False
        has = True
    if "$lte" in ops:
        hi, hi_inc, has = ops["$lte"], True, True
    if "$lt" in ops:
        val = ops["$lt"]
        if hi is None or _cmpv(val, hi) < 0:
            hi, hi_inc = val, False
        has = True
    return (lo, hi, lo_inc, hi_inc) if has else None


def _cmpv(a: Any, b: Any) -> int:
    ka, kb = V.encode_key(a), V.encode_key(b)
    return (ka > kb) - (ka < kb)


@dataclass
class _Branch:
    prefix: Tuple[tuple, ...] = ()
    lo: Optional[tuple] = None       # encoded interval low at current clause
    hi: Optional[tuple] = None
    lo_inc: bool = True
    hi_inc: bool = True
    closed: bool = False             # a range consumed this branch
    dead: bool = False               # no possible key on this branch


def _branch_to_range(b: _Branch, n: int) -> Optional[KeyRange]:
    if b.dead:
        return None
    k = len(b.prefix)
    if b.closed:
        lo = b.prefix + ((b.lo,) if b.lo is not None else (V.MINKEY,))
        hi = b.prefix + ((b.hi,) if b.hi is not None else (V.MAXKEY,))
        lo += (V.MINKEY,) * (n - len(lo))
        hi += (V.MAXKEY,) * (n - len(hi))
        return KeyRange(lo=lo, hi=hi,
                        lo_inclusive=b.lo_inc if b.lo is not None else True,
                        hi_inclusive=b.hi_inc if b.hi is not None else True)
    lo = b.prefix + (V.MINKEY,) * (n - k)
    hi = b.prefix + (V.MAXKEY,) * (n - k)
    return KeyRange(lo=lo, hi=hi, lo_inclusive=True, hi_inclusive=True)


class _PlanAbort(Exception):
    pass


def build_ranges(idef: IndexDef, preds: List[Pred], multikey: bool
                 ) -> Tuple[List[KeyRange], List[Pred], List[str]]:
    """Choose ONE element group and build key ranges from it.

    The invariant that makes a compound multikey scan correct: every key
    component we use to narrow the scan must describe the *same* array element.
    So we pick a single group of predicates that all share an element scope,
    and then take the longest index-clause prefix all of whose predicates are
    in that group.  Predicates in any other group stay residual; the matcher
    re-checks them.  On a non-multikey index there is one element per document,
    so every predicate can share the group.

    Returns (ranges, used_predicates, notes).  The union of ranges is a
    superset of every matching document that owns a key on this index.
    """
    clauses = idef.clauses
    n = len(clauses)
    by_field: Dict[str, List[Pred]] = {}
    for p in preds:
        by_field.setdefault(p.field, []).append(p)

    groups = _element_groups(preds, multikey)
    per_group: List[Tuple[int, List[KeyRange], List[Pred], List[str]]] = []
    all_notes: List[str] = []
    for gid, group in groups:
        try:
            ranges, used, notes = _ranges_for_group(
                idef, group, by_field, multikey, gid)
        except _PlanAbort:
            continue
        if not ranges:
            continue
        score = (len(used),
                 sum(1 for p in used if "$eq" in p.ops or "$in" in p.ops),
                 -len(ranges))
        per_group.append((score, ranges, used, notes))
    if not per_group:
        return [], [], all_notes or ["no element group yields a usable key range"]
    per_group.sort(key=lambda t: t[0], reverse=True)
    best = per_group[0]
    # If several witness groups constrain the SAME leading clause, union their
    # ranges (a matching document owns at least one of those keys).  This is a
    # tighter scan than picking one witness and keeping the rest residual.
    if len(per_group) > 1:
        best = _maybe_union_witness_groups(idef, per_group, best)
    _score, ranges, used, notes = best
    used_ids = {id(p) for p in used}
    skipped = [p for p in preds if id(p) not in used_ids]
    if skipped:
        notes.append(
            "predicates left for the matcher because they describe a different "
            "element scope: " +
            ", ".join(sorted({f"{p.field} {sorted(p.ops)}" for p in skipped})))
    return ranges, used, notes


def _maybe_union_witness_groups(idef: IndexDef, per_group, best):
    """Union ranges from groups that all constrain clause 0 the same way."""
    primary_fields = {idef.clauses[0].field}
    merged_ranges: List[KeyRange] = []
    merged_used: List[Pred] = []
    notes: List[str] = []
    all_single_clause = True
    for score, ranges, used, nts in per_group:
        if not used or any(p.field not in primary_fields for p in used):
            all_single_clause = False
            break
        merged_ranges.extend(ranges)
        merged_used.extend(used)
        notes.extend(nts)
    if not all_single_clause or len(merged_ranges) <= len(best[1]):
        return best
    # dedupe
    seen = set()
    uniq = []
    for r in merged_ranges:
        sig = (r.lo, r.hi, r.lo_inclusive, r.hi_inclusive)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)
    notes = list(dict.fromkeys(notes))
    notes.append(
        f"clause 0 ({idef.clauses[0].field!r}): {len(per_group)} independent witness "
        f"groups unioned into {len(uniq)} ranges -- each group's range is a superset "
        f"of the documents it can satisfy, and a document need satisfy only one")
    return ((len(merged_used), 0, -len(uniq)), uniq, merged_used, notes)


def _element_groups(preds: List[Pred], multikey: bool
                    ) -> List[Tuple[str, List[Pred]]]:
    if not multikey:
        return [("all", list(preds))]
    groups: Dict[str, List[Pred]] = {}
    for p in preds:
        if p.scope == "same-element" and p.align_root is not None:
            key = f"elemMatch({p.align_root})"
        else:
            key = f"witness({p.field},#{id(p)})"
        groups.setdefault(key, []).append(p)
    # prefer groups with more predicates and with lower selectivity risk
    items = list(groups.items())
    items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    return items


def _ranges_for_group(idef: IndexDef, group: List[Pred],
                      by_field: Dict[str, List[Pred]], multikey: bool,
                      gid: str) -> Tuple[List[KeyRange], List[Pred], List[str]]:
    in_group = {id(p) for p in group}
    group_by_field: Dict[str, List[Pred]] = {}
    for p in group:
        group_by_field.setdefault(p.field, []).append(p)
    # fields that have predicates but none in this group: cannot continue past
    # such a clause, because the next key component would describe an element
    # this group did not choose
    excluded_fields = {p.field for p in by_field_value(by_field)
                       if all(id(q) not in in_group for q in by_field[p.field])}

    branches: List[_Branch] = [_Branch()]
    used: List[Pred] = []
    notes: List[str] = [f"element scope: {gid}"]

    for i, clause in enumerate(idef.clauses):
        f = clause.field
        if f in excluded_fields:
            notes.append(
                f"clause {i} ({f!r}) not used: it only has predicates from a "
                f"different element scope than the chosen key components")
            break
        cand = group_by_field.get(f, [])
        if not cand:
            if i == 0:
                raise _PlanAbort()
            notes.append(f"clause {i} ({f!r}) not used: no predicate; prefix ends")
            break
        if all(b.closed for b in branches):
            notes.append(f"clause {i} ({f!r}) not used: an earlier range fixed the "
                         f"prefix")
            break

        # Split this clause's predicates into "equality" and "range" shapes,
        # but ALSO into independent witnesses.  Two separate `$gt`/`$lt`
        # predicates on a multikey field are separate witnesses: each is turned
        # into its own range and the ranges are unioned.  Only predicates that
        # share one element (one $elemMatch body, or a non-multikey index) may
        # contribute to a single intersected component.
        eqs: List[Any] = []
        intervals: List[Tuple[Any, Any, bool, bool]] = []
        this_used: List[Pred] = []
        witness_max = 1  # when independent witnesses exist, cap the prefix here
        for p in cand:
            pops = p.element_ops
            pts = _eq_points(pops)
            rng = _range_interval(pops)
            if pts is not None:
                if rng is not None:
                    pts = [v for v in pts if _in_interval(v, rng)]
                if pts:
                    eqs.extend(pts)
                    this_used.append(p)
                continue
            if rng is not None:
                intervals.append(rng)
                this_used.append(p)
        eqs = _dedupe(eqs)

        # Everything in one group is guaranteed to share an element, so all
        # intervals may be intersected.
        # Are these predicates one element scope or several independent ones?
        # On a non-multikey index there is exactly one value per document, so
        # every predicate describes that same value: intersect them.
        if multikey:
            n_scopes = len({p.scope if p.scope == "same-element" else f"w{id(p)}"
                            for p in cand})
        else:
            n_scopes = 1
        if n_scopes > 1:
            # Independent witnesses: document must have SOME element for each.
            # Their key sets are unioned (a doc may be reached through any one
            # witness), so we must scan the union of every witness's range.
            branches = _fork_by_witness(branches, cand)
            used.extend(this_used)
            notes.append(
                f"clause {i} ({f!r}): {n_scopes} independent element witnesses; "
                f"each predicate contributes its own range and the ranges are "
                f"UNIONED, never intersected (intersection would drop documents "
                f"whose different elements satisfy the different predicates)")
            # every remaining clause is residual now
            break
        if eqs:
            if intervals:
                merged = _combine(intervals)
                eqs = [v for v in eqs if _in_interval(v, merged)]
                if len(intervals) + len(eqs) > 1:
                    notes.append(
                        f"clause {i} ({f!r}): equality values and range bounds "
                        f"come from the same element scope; intersected")
            if eqs:
                if len(intervals) > 1:
                    notes.append(
                        f"clause {i} ({f!r}): {len(intervals)} range bounds MERGED "
                        f"into one interval (same element guaranteed)")
                branches = _fork_eq(branches, eqs)
                used.extend(this_used)
                continue
            notes.append(f"clause {i} ({f!r}): equality incompatible with the "
                         f"range on the same element; range only")
        if intervals:
            if len(intervals) > 1:
                notes.append(f"clause {i} ({f!r}): {len(intervals)} range bounds "
                             f"MERGED into one interval (same element guaranteed)")
            elif _has_two_sided(intervals[0]):
                notes.append(f"clause {i} ({f!r}): lower and upper bound MERGED "
                             f"into one interval (same element guaranteed)")
            else:
                notes.append(f"clause {i} ({f!r}): range bound")
            branches = _fork_range(branches, _combine(intervals))
            used.extend(this_used)
            continue
        if not any(p.element_ops for p in cand):
            notes.append(f"clause {i} ({f!r}): only residual operators "
                         f"({sorted(cand[0].ops)})")
        break

    ranges: List[KeyRange] = []
    seen = set()
    for b in branches:
        r = _branch_to_range(b, idef and len(idef.clauses))
        if r is None or not _range_nonempty(r):
            continue
        sig = (r.lo, r.hi, r.lo_inclusive, r.hi_inclusive)
        if sig in seen:
            continue
        seen.add(sig)
        ranges.append(r)
    return ranges, used, notes


def by_field_value(by_field: Dict[str, List[Pred]]) -> List[Pred]:
    out: List[Pred] = []
    for v in by_field.values():
        out.extend(v)
    return out


def _range_nonempty(r: KeyRange) -> bool:
    if r.lo is None or r.hi is None:
        return True
    if r.lo < r.hi:
        return True
    if r.lo == r.hi:
        return r.lo_inclusive and r.hi_inclusive
    return False


def _combine(intervals: List[Tuple[Any, Any, bool, bool]]
             ) -> Tuple[Any, Any, bool, bool]:
    out = intervals[0]
    for iv in intervals[1:]:
        out = _intersect(out, iv)
    return out


def _dedupe(vals: List[Any]) -> List[Any]:
    out, seen = [], set()
    for v in vals:
        k = V.encode_key(v)
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _in_interval(v: Any, interval: Tuple[Any, Any, bool, bool]) -> bool:
    lo, hi, lo_inc, hi_inc = interval
    kv = V.encode_key(v)
    if lo is not None:
        kl = V.encode_key(lo)
        if kv < kl or (kv == kl and not lo_inc):
            return False
    if hi is not None:
        kh = V.encode_key(hi)
        if kv > kh or (kv == kh and not hi_inc):
            return False
    return True


def _intersect(prev: Optional[Tuple[Any, Any, bool, bool]],
               cur: Tuple[Any, Any, bool, bool]) -> Tuple[Any, Any, bool, bool]:
    if prev is None:
        return cur
    plo, phi, pli, phi_ = prev
    clo, chi, cli, chi_ = cur
    lo, lo_inc = plo, pli
    if clo is not None:
        if lo is None or _cmpv(clo, lo) > 0:
            lo, lo_inc = clo, cli
        elif _cmpv(clo, lo) == 0:
            lo_inc = cli and pli
    hi, hi_inc = phi, phi_
    if chi is not None:
        if hi is None or _cmpv(chi, hi) < 0:
            hi, hi_inc = chi, chi_
        elif _cmpv(chi, hi) == 0:
            hi_inc = chi_ and phi_
    return (lo, hi, lo_inc, hi_inc)


def _fork_eq(branches: List[_Branch], vals: List[Any]) -> List[_Branch]:
    out = []
    for b in branches:
        if b.closed:
            # equality still narrows a closed branch only if key-level equal to
            # the fixed interval's single point; simplest safe answer is to keep
            # the branch unchanged (matcher re-checks) rather than discard it.
            out.append(b)
            continue
        for v in vals:
            out.append(_Branch(prefix=b.prefix + (_enc(v),), closed=False))
    return out


def _fork_range(branches: List[_Branch],
                interval: Tuple[Any, Any, bool, bool]) -> List[_Branch]:
    lo, hi, lo_inc, hi_inc = interval
    out = []
    for b in branches:
        if b.closed or b.dead:
            continue
        if lo is not None and hi is not None and _cmpv(lo, hi) > 0:
            continue
        nb = _Branch(prefix=b.prefix,
                     lo=_enc(lo) if lo is not None else None,
                     hi=_enc(hi) if hi is not None else None,
                     lo_inc=lo_inc, hi_inc=hi_inc, closed=True)
        if not _branch_nonempty(nb):
            continue
        out.append(nb)
    return out


def _fork_each_range(branches: List[_Branch],
                     intervals: List[Tuple[Any, Any, bool, bool]]) -> List[_Branch]:
    out = []
    for b in branches:
        if b.closed or b.dead:
            continue
        for iv in intervals:
            out.extend(_fork_range([b], iv))
    return out


def _has_two_sided(iv: Tuple[Any, Any, bool, bool]) -> bool:
    return iv[0] is not None and iv[1] is not None


def _fork_by_witness(branches: List[_Branch], preds: List[Pred]) -> List[_Branch]:
    """One branch per predicate, unioned: the safe reading for independent
    multikey witnesses.  Each predicate's own ops are intersected internally
    (they describe one element)."""
    out: List[_Branch] = []
    for b in branches:
        if b.closed or b.dead:
            continue
        for p in preds:
            pops = p.element_ops
            pts = _eq_points(pops)
            rng = _range_interval(pops)
            if pts is not None:
                if rng is not None:
                    pts = [v for v in pts if _in_interval(v, rng)]
                for v in _dedupe(pts or []):
                    out.extend(_fork_eq([b], [v]))
                continue
            if rng is not None:
                out.extend(_fork_range([b], rng))
    # fall back to the whole index when nothing contributed a usable bound
    return out if out else branches


def _branch_nonempty(b: _Branch) -> bool:
    if b.lo is None or b.hi is None:
        return True
    if b.lo < b.hi:
        return True
    if b.lo == b.hi:
        return b.lo_inc and b.hi_inc
    return False


# --------------------------------------------------------------------------
# Plan objects
# --------------------------------------------------------------------------

@dataclass
class ScanLeg:
    index: Index
    ranges: List[KeyRange]
    used_predicates: List[Pred]
    notes: List[str]
    direction: int = 1
    kind: str = "range"


@dataclass
class Plan:
    mode: str
    legs: List[ScanLeg] = field(default_factory=list)
    tree: Any = None
    residual_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    sort_in_memory: bool = False
    sort: List[Tuple[str, int]] = field(default_factory=list)
    # execution counters
    keys_examined: int = 0
    postings_read: int = 0
    candidates: int = 0
    docs_fetched: int = 0
    docs_matched: int = 0
    docs_returned: int = 0
    elapsed_ms: float = 0.0

    @property
    def index_names(self) -> List[str]:
        seen: List[str] = []
        for leg in self.legs:
            if leg.index.name not in seen:
                seen.append(leg.index.name)
        return seen

    def to_dict(self) -> dict:
        legs = []
        for leg in self.legs:
            legs.append({
                "index": leg.index.name,
                "fields": leg.index.defn.fields,
                "multikey": leg.index.defn.multikey_declared,
                "kind": leg.kind,
                "direction": leg.direction,
                "ranges": [r.describe(len(leg.index.defn.clauses))
                           for r in leg.ranges],
                "predicates": [f"{p.field} {sorted(p.ops)}" for p in leg.used_predicates],
                "notes": leg.notes,
            })
        return {
            "mode": self.mode,
            "indexes": self.index_names,
            "legs": legs,
            "residual_fields": self.residual_fields,
            "notes": self.notes,
            "sort": [{"field": f, "direction": d} for f, d in self.sort],
            "sort_in_memory": self.sort_in_memory,
            "counters": {
                "index_keys_examined": self.keys_examined,
                "index_postings_read": self.postings_read,
                "candidates": self.candidates,
                "documents_fetched": self.docs_fetched,
                "documents_matched": self.docs_matched,
                "documents_returned": self.docs_returned,
                "elapsed_ms": self.elapsed_ms,
            },
        }

    def text(self) -> str:
        lines = [f"mode: {self.mode}"]
        for leg in self.legs:
            lines.append(f"  index {leg.index.name} on {leg.index.defn.fields} "
                         f"(multikey={leg.index.defn.multikey_declared})")
            for r in leg.ranges:
                lines.append(f"    scan {r.describe(len(leg.index.defn.clauses))}")
            for nt in leg.notes:
                lines.append(f"    - {nt}")
        if self.residual_fields:
            lines.append("  residual (matcher only): " + ", ".join(self.residual_fields))
        if self.sort_in_memory and self.sort:
            lines.append("  sort: in memory")
        for nt in self.notes:
            lines.append(f"  note: {nt}")
        lines.append(
            f"  counters: keys={self.keys_examined} postings={self.postings_read} "
            f"candidates={self.candidates} fetched={self.docs_fetched} "
            f"matched={self.docs_matched} returned={self.docs_returned} "
            f"elapsed={self.elapsed_ms:.2f}ms")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

SORT_OP_ONLY = {"$eq", "$gt", "$gte", "$lt", "$lte", "$in"}


def _estimate_leg_docs(leg: ScanLeg) -> int:
    """Estimate *documents* a leg will fetch.

    Multikey indexes break the simple "documents / distinct keys" average: a
    single-field size index has 10 distinct keys but each is in ~30% of the
    documents.  So estimate from selectivity instead -- how much of a clause's
    value domain the range covers.  The estimate may be wrong; it is only used
    to choose between plans, never to decide results.
    """
    idx = leg.index
    total_docs = max(1, len(idx.doc_keys))
    if not leg.ranges:
        return total_docs
    if any(r.lo is None and r.hi is None for r in leg.ranges):
        return total_docs

    # selectivity of the leading clause, per range
    n_distinct = _distinct_key_count(idx)
    selectivity = 0.0
    for r in leg.ranges:
        if _is_equality_range(r):
            # one value out of the leading clause's domain
            selectivity += 1.0 / max(1, n_distinct)
        else:
            # unknown numeric/string span: assume a quarter of the domain,
            # tightened a little when the domain is large
            selectivity += 0.25 if n_distinct <= 64 else 0.12
    selectivity = min(1.0, selectivity)
    # a document is fetched once even if several of its elements match, so the
    # doc count is what matters; multikey only makes the fetch *more* likely
    # than the raw selectivity suggests, so bias upward slightly
    est = int(total_docs * selectivity)
    return max(1, min(total_docs, est))


def _is_equality_range(r: KeyRange) -> bool:
    return (r.lo is not None and r.hi is not None and
            r.lo[:-1] == r.hi[:-1] and
            (r.lo[-1] == r.hi[-1] or
             (len(r.lo) and r.lo[-1] == V.MINKEY and r.hi[-1] == V.MAXKEY)))


def _distinct_key_count(idx: Index) -> int:
    return max(1, sum(1 for p in idx.postings.values() if p))


def _leg_cost(leg: ScanLeg) -> int:
    """Cost in document fetches plus a small per-key overhead."""
    return _estimate_leg_docs(leg) + 2 * max(1, len(leg.ranges))


def _leg_rows(leg: ScanLeg) -> int:
    """Estimated posting rows a leg reads (what dominates an intersection)."""
    idx = leg.index
    if not leg.ranges:
        return max(1, idx.entry_count)
    if any(r.lo is None and r.hi is None for r in leg.ranges):
        return max(1, idx.entry_count)
    n_distinct = _distinct_key_count(idx)
    per_key = max(1, idx.entry_count // n_distinct)
    est = 0
    for r in leg.ranges:
        if _is_equality_range(r):
            est += per_key
        else:
            est += max(1, idx.entry_count // 4)
    return max(1, min(idx.entry_count, est))


def _leg_total_cost(leg: ScanLeg) -> int:
    """Rows read plus documents fetched: the number that matters for timing."""
    return _leg_rows(leg) + _estimate_leg_docs(leg)


def choose_index(indexes: Sequence[Index], preds: List[Pred]
                 ) -> Tuple[Optional[Index], List[KeyRange], List[Pred], List[str], int]:
    best = None
    for idx in indexes:
        multikey = idx.defn.multikey_declared or idx.multikey_docs > 0
        ranges, used, notes = build_ranges(idx.defn, preds, multikey)
        if not used or not ranges:
            continue
        equality_clauses = 0
        for p in used:
            if "$eq" in p.ops or "$in" in p.ops:
                equality_clauses += 1
        leg = ScanLeg(index=idx, ranges=ranges, used_predicates=used, notes=notes)
        cost = _leg_cost(leg)
        score = (equality_clauses, len(used), -cost, -len(ranges))
        if best is None or score > best[0]:
            best = (score, idx, ranges, used, notes, cost)
    if best is None:
        return None, [], [], [], 0
    score, idx, ranges, used, notes, cost = best
    return idx, ranges, used, notes, cost


def _total_docs(indexes: Sequence[Index], store=None) -> int:
    if store is not None:
        return max(1, store.count())
    if indexes:
        return max(1, len(indexes[0].doc_keys))
    return 1


def _collscan_cost(total_docs: int) -> int:
    return max(1, total_docs)


def build_plan(tree: Any, indexes: Sequence[Index],
               sort: Sequence[Tuple[str, int]], store=None) -> Plan:
    preds, gather_notes = gather_predicates(tree)
    idx, ranges, used, notes, ix_cost = choose_index(indexes, preds)
    total_docs = _total_docs(indexes, store)
    scan_cost = _collscan_cost(total_docs)

    plan = Plan(mode="collscan", tree=tree)
    plan.notes.extend(gather_notes)
    plan.sort = list(sort)

    used_fields = {p.field for p in used}
    residual = sorted({p.field for p in preds if p.field not in used_fields})
    plan.residual_fields = residual

    design_idx = idx
    # Cost comparison only matters at scale; for small collections an index
    # scan is always at least as good and keeps plan output meaningful.  The
    # cost used here includes posting rows *read*, which is what actually costs
    # time when a range on a low-cardinality multikey index touches most rows.
    if idx is not None and total_docs >= 500:
        ix_total = _leg_total_cost(plan.legs[0] if plan.legs else ScanLeg(
            idx, ranges, used, notes))
        if ix_total >= scan_cost:
            plan.notes.append(
                f"index {idx.name!r} would read about {ix_total:,} index rows and "
                f"documents versus {scan_cost:,} documents in a collection scan; "
                f"the predicate is not selective enough for this index, so a "
                f"collection scan is used")
            idx = None

    if idx is not None:
        leg = ScanLeg(index=idx, ranges=ranges, used_predicates=used, notes=notes)
        plan.mode = "ixscan"
        plan.legs = [leg]
        # If several predicates are still residual, intersecting per-predicate
        # legs can narrow the candidate set a lot (especially for independent
        # element semantics on a multikey index, where one leg cannot combine
        # clauses).  Cost of an intersection is bounded by the *largest* leg,
        # not their sum: postings are intersected by doc id before fetching.
        if len(used) < len(preds):
            legs2 = _intersection_legs(indexes, preds)
            if len(legs2) >= 2:
                # Intersection reads every leg's postings, then fetches only the
                # survivors.  The fetch count is bounded below by the smallest
                # leg and above by the smallest-leg estimate; use the smallest
                # leg's cost plus a small per-extra-leg overhead.
                inter_cost = (sum(_leg_rows(l) for l in legs2) +
                              min(_leg_cost(l) for l in legs2))
                if total_docs < 500 or inter_cost < ix_cost:
                    plan.mode = "index_intersection"
                    plan.legs = legs2
                    plan.notes.append(
                        "index intersection chosen: a multikey compound index "
                        "cannot combine independently-satisfied clauses into one "
                        "key, so per-predicate scans are intersected by document "
                        "id (cost bounded by the largest leg); the residual "
                        "matcher still confirms every row")
    else:
        legs = _intersection_legs(indexes, preds)
        if legs and len(legs) >= 2:
            # Intersection reads every leg's postings, then fetches the
            # survivors once.  Its cost is therefore the posting rows read plus
            # the smallest leg's document estimate, not the sum of doc fetches.
            rows = sum(_leg_rows(l) for l in legs)
            survivors = min(_estimate_leg_docs(l) for l in legs)
            inter_total = rows + survivors
        else:
            inter_total = scan_cost + 1
        single_total = _leg_total_cost(ScanLeg(idx, ranges, used, notes)) if idx else scan_cost
        if legs and len(legs) >= 2 and (total_docs < 500 or inter_total < min(single_total, scan_cost)):
            plan.mode = "index_intersection"
            plan.legs = legs
            plan.notes.append(
                "index intersection: no single index covers enough predicates, so "
                "per-predicate legs are intersected by document id; the residual "
                "matcher confirms every surviving row")
        elif idx is None and legs and len(legs) >= 2:
            plan.mode = "index_intersection"
            plan.legs = legs
            plan.notes.append("index intersection chosen over a collection scan")
        else:
            plan.mode = "collscan"

    if sort:
        if plan.mode in ("ixscan",) and _sort_covered(plan.legs[0], sort):
            leg = plan.legs[0]
            leg.direction = _sort_direction(leg.index, sort)
            plan.sort_in_memory = False
        elif plan.mode == "collscan" and design_idx is not None and \
                _sort_covered(ScanLeg(design_idx, [], [], []), sort):
            # Even though a scan is cheaper for filtering, an index can supply
            # the order without sorting in memory; use it as an ordered scan
            # only if that does not defeat the cost choice.
            plan.sort_in_memory = True
        else:
            plan.sort_in_memory = True

    if plan.mode == "collscan":
        plan.notes.append("full collection scan with the reference matcher")
    return plan


def _intersection_legs(indexes: Sequence[Index], preds: List[Pred]) -> List[ScanLeg]:
    """Build one leg per *predicate group* for index intersection.

    Different predicate groups are intersected; predicates inside one group are
    unioned.  The grouping rule matches the semantics:

    * independent-element predicates each get their own group, because they
      may be satisfied by different elements (union across a group is wrong,
      but intersecting different fields is right);
    * a single ``$elemMatch`` body is one group, since one element satisfies
      all of them (they can share an aligned compound key).

    Each leg scans the index whose *leading clause* is the predicate's field.
    """
    groups: List[List[Pred]] = []
    for p in preds:
        if p.scope == "same-element" and p.align_root is not None:
            for g in groups:
                if g and g[0].scope == "same-element" and \
                        g[0].align_root == p.align_root:
                    g.append(p)
                    break
            else:
                groups.append([p])
        else:
            groups.append([p])

    legs: List[ScanLeg] = []
    for group in groups:
        best = None
        for idx in indexes:
            leading = idx.defn.clauses[0].field
            if not any(p.field == leading for p in group):
                continue
            # if the group's fields differ, only the ones matching the leading
            # clause shape this leg
            sub = [p for p in group if p.field == leading]
            multikey = idx.defn.multikey_declared or idx.multikey_docs > 0
            ranges, used, notes = build_ranges(idx.defn, sub, multikey)
            if not used or not ranges:
                continue
            cost = _leg_cost(ScanLeg(idx, ranges, used, notes))
            if best is None or cost < best[0]:
                best = (cost, ScanLeg(idx, ranges, used,
                                      notes + ["intersection leg for " +
                                               ", ".join(sorted({p.field for p in used}))]))
        if best is not None:
            legs.append(best[1])
    return legs


def _sort_covered(leg: ScanLeg, sort: Sequence[Tuple[str, int]]) -> bool:
    fields = leg.index.defn.fields
    if len(sort) > len(fields):
        return False
    for i, (f, d) in enumerate(sort):
        if fields[i] != f:
            return False
    dirs = leg.index.defn.directions
    want = [d for _, d in sort]
    got = dirs[:len(sort)]
    if all((w > 0) == (g > 0) for w, g in zip(want, got)):
        return True
    if all((w > 0) != (g > 0) for w, g in zip(want, got)):
        return True
    return False


def _sort_direction(leg_or_idef, sort: Sequence[Tuple[str, int]]) -> int:
    idef = leg_or_idef.defn if hasattr(leg_or_idef, "defn") else leg_or_idef
    want = [d for _, d in sort]
    got = idef.directions[:len(sort)]
    if all((w > 0) == (g > 0) for w, g in zip(want, got)):
        return 1
    return -1


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def execute(plan: Plan, store, projection: Optional[Dict[str, Any]] = None,
            limit: int = 0, skip: int = 0) -> Tuple[List[dict], Plan]:
    import time as _time
    start = _time.perf_counter()
    tree = plan.tree

    candidate_ids: Optional[List[Any]] = None

    if plan.mode in ("ixscan", "index_intersection"):
        legs = list(plan.legs)
        if plan.mode == "ixscan" and not plan.sort_in_memory and plan.legs:
            # index-ordered scan
            ordered_ids = _ordered_scan(plan.legs[0], plan)
            candidate_ids = ordered_ids
            plan.candidates = len(candidate_ids or [])
    if plan.mode in ("ixscan", "index_intersection") and candidate_ids is None:
        per_leg: List[List[Any]] = []
        for leg in plan.legs:
            ids, keys, postings = _run_leg(leg)
            plan.keys_examined += keys
            plan.postings_read += postings
            per_leg.append(ids)
        if len(per_leg) == 1:
            candidate_ids = per_leg[0]
        else:
            sets = [set(x) for x in per_leg]
            base = min(sets, key=len)
            merged = [i for i in per_leg[0] if all(i in s for s in sets)]
            candidate_ids = merged
        plan.candidates = len(candidate_ids)

    docs_out: List[dict] = []
    if candidate_ids is None:
        # collection scan
        for doc in store.scan():
            plan.docs_fetched += 1
            if matches(doc, tree):
                plan.docs_matched += 1
                docs_out.append(doc)
        if not plan.sort_in_memory and not plan.sort:
            # deterministic default order, identical to the indexed path
            docs_out.sort(key=lambda d: V.encode_key(d.get("_id")))
    else:
        # Candidate order is normalised to encoded _id order so that the indexed
        # path and the full-scan path return documents in the same sequence
        # (which is what "row by row" comparison relies on).  With an explicit
        # sort the sorted order is applied below; the index-ordered scan keeps
        # its own order.
        ordered = candidate_ids
        if not (plan.mode == "ixscan" and not plan.sort_in_memory and plan.sort):
            ordered = sorted(candidate_ids, key=lambda i: V.encode_key(i))
        for doc_id in ordered:
            doc = store.get(doc_id)
            if doc is None:
                continue
            plan.docs_fetched += 1
            if matches(doc, tree):
                plan.docs_matched += 1
                docs_out.append(doc)

    if plan.sort_in_memory and plan.sort:
        docs_out.sort(key=lambda d: _sort_key(d, plan.sort))
    elif plan.mode == "collscan" and plan.sort:
        docs_out.sort(key=lambda d: _sort_key(d, plan.sort))

    if skip:
        docs_out = docs_out[skip:]
    if limit:
        docs_out = docs_out[:limit]
    plan.docs_returned = len(docs_out)
    if projection is not None:
        docs_out = [project(d, projection) for d in docs_out]
    plan.elapsed_ms = (_time.perf_counter() - start) * 1000.0
    return docs_out, plan


def _run_leg(leg: ScanLeg) -> Tuple[List[Any], int, int]:
    ids: List[Any] = []
    seen: Set[Any] = set()
    keys_total = 0
    postings_total = 0
    for r in leg.ranges:
        got, keys, postings = leg.index.scan_range(
            r.lo, r.hi, r.lo_inclusive, r.hi_inclusive)
        keys_total += keys
        postings_total += postings
        for d in got:
            if d not in seen:
                seen.add(d)
                ids.append(d)
    leg.kind = "intersection-leg"
    return ids, keys_total, postings_total


def _ordered_scan(leg: ScanLeg, plan: Plan) -> List[Any]:
    """Return docs in index order (deduped), honouring leg.direction.

    Documents that share a key are ordered by encoded ``_id`` so that an
    index-supplied order and an in-memory sort (which breaks ties the same
    way) produce identical sequences.
    """
    idx = leg.index
    wanted: List[tuple] = []
    for r in leg.ranges:
        for k in idx.keys.range_keys(r.lo, r.hi, r.lo_inclusive, r.hi_inclusive):
            wanted.append(k)
    order = sorted(set(wanted))
    if leg.direction < 0:
        order = list(reversed(order))
    ids: List[Any] = []
    seen: Set[Any] = set()
    for k in order:
        postings = idx.postings.get(k)
        if not postings:
            continue
        plan.keys_examined += 1
        plan.postings_read += len(postings)
        for d in sorted(postings, key=lambda i: V.encode_key(i)):
            if d not in seen:
                seen.add(d)
                ids.append(d)
    leg.kind = "ordered-range"
    return ids


def _sort_key(doc: dict, sort: Sequence[Tuple[str, int]]):
    """Sort key for in-memory ordering.

    Array fields sort by their smallest element ascending, or largest element
    descending (documented).  Python's sort is stable and ascending only, so a
    descending field is encoded by wrapping each component in a
    ``_Descending`` marker that inverts comparison order for that component
    only -- this keeps mixed-direction multi-field sorts correct.
    """
    from .paths import get_path
    from . import values as V
    key = []
    for f, direction in sort:
        matched = get_path(doc, f)
        if not matched:
            comp = V.MISSING_KEY
        else:
            vals = []
            for v in matched:
                if isinstance(v, list):
                    vals.extend(v)
                else:
                    vals.append(v)
            if not vals:
                comp = V.EMPTY_ARRAY_KEY  # empty array sorts with arrays, not missing
            else:
                enc = [V.encode_key(v) for v in vals]
                comp = min(enc) if direction > 0 else max(enc)
        key.append(_Descending(comp) if direction < 0 else comp)
    key.append(V.encode_key(doc.get("_id")))
    return tuple(key)


class _Descending:
    """Wraps a sort component so comparisons run in the opposite order."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value > other.value

    def __gt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value >= other.value

    def __ge__(self, other):
        return self.value <= other.value

    def __hash__(self):
        return hash(self.value)


# --------------------------------------------------------------------------
# Covering analysis
# --------------------------------------------------------------------------

def covering_report(plan: Plan, projection: Optional[Dict[str, Any]],
                    query_fields: Sequence[str]) -> dict:
    blockers: List[str] = []
    if plan.mode == "collscan":
        blockers.append("collection scan reads full documents")
    if plan.mode == "index_intersection":
        blockers.append("index intersection needs document fetches to combine keys")
    if plan.legs:
        if any(leg.index.defn.multikey_declared or leg.index.multikey_docs > 0
               for leg in plan.legs):
            blockers.append(
                "multikey index: one document owns many keys, so the key cannot "
                "reconstruct the document's field values unambiguously")
        idx_fields = set(plan.legs[0].index.defn.fields)
        missing = sorted(set(query_fields) - idx_fields - {"_id"})
        if missing:
            blockers.append(f"query references fields not in the index: {missing}")
        if plan.residual_fields:
            blockers.append(
                f"residual predicates checked on the document: {plan.residual_fields}")
        if projection is not None:
            proj_fields = _projection_fields(projection)
            missing_p = sorted(proj_fields - idx_fields - {"_id"})
            if missing_p:
                blockers.append(
                    f"projection needs fields outside the index: {missing_p}")
        elif projection is None:
            blockers.append("no projection: the full document is required")
        if plan.sort_in_memory and plan.sort:
            blockers.append("sort is not supplied by the index")
    else:
        blockers.append("no index is used")
    return {
        "possible": not blockers,
        "blockers": blockers,
        "note": ("the engine always re-fetches documents so that returned values "
                 "are byte-for-byte the stored values, even if a covering return "
                 "were possible"),
    }


def _projection_fields(projection: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    include = True
    for k, v in projection.items():
        if k == "_id":
            continue
        if v in (0, False):
            include = False
    for k, v in projection.items():
        if k == "_id":
            continue
        if include and v not in (0, False):
            out.add(k)
        elif not include and v not in (0, False):
            out.add(k)
    return out


def project(doc: dict, projection: Dict[str, Any]) -> dict:
    from .paths import get_path_one
    if not projection:
        return dict(doc)
    include = {k: v for k, v in projection.items() if v not in (0, False)}
    exclude = {k: v for k, v in projection.items() if v in (0, False)}
    if include and exclude and "_id" not in exclude:
        # mixed: treat positive fields + drop excluded (documented simple rule)
        pass
    if include:
        out: Dict[str, Any] = {}
        if include.get("_id", 1) not in (0, False) or "_id" not in projection:
            if "_id" in doc:
                out["_id"] = doc["_id"]
        for k in include:
            if k == "_id":
                continue
            ok, val = get_path_one(doc, k)
            if ok:
                out[k] = val
        return out
    out = dict(doc)
    for k in exclude:
        if k == "_id":
            out.pop("_id", None)
        else:
            out.pop(k, None)
    return out
