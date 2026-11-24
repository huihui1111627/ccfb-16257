"""Deterministic catalogue generator for the benchmark.

Produces ``N`` product documents covering the awkward shapes on purpose:

* spec arrays of length 0..8 (lengths vary, long tail)
* empty arrays     ``"specs": []``
* missing fields   no ``specs`` key at all; specs with no ``size``/``stock``
* mixed types      ``specs`` as a string or object; ``price`` as a string;
                   ``stock`` as a bool or float; ``size`` as an int
* top-level scalars the ops query does not use, plus a ``brand`` and ``rating``
  that a couple of indexes reuse
* a small set of "trap" documents whose colour+size+stock are spread across
  different elements -- the exact documents the buggy query wrongly returned

The generator is seeded, so a run is reproducible from ``--seed``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator

COLORS = ["red", "blue", "green", "black", "white", "pink", "navy", "grey"]
COLOR_WEIGHTS = [30, 18, 12, 12, 10, 7, 6, 5]
SIZES = ["S", "M", "L", "XL", "XXL", "3XL"]
SIZE_WEIGHTS = [18, 22, 22, 22, 10, 6]
BRANDS = ["acme", "northwind", "globex", "initech", "umbrella", "hooli"]
SEASONS = ["2024SS", "2024AW", "2025SS", "2025AW"]
CATEGORIES = ["shirt", "jacket", "pants", "dress", "shoes", "hat"]
MATERIALS = ["cotton", "wool", "polyester", "silk", "leather"]


def spec_element(rng: random.Random, *, weird: bool = False) -> Dict[str, Any]:
    el: Dict[str, Any] = {}
    el["color"] = rng.choices(COLORS, COLOR_WEIGHTS)[0]
    if rng.random() < 0.97:
        el["size"] = rng.choices(SIZES, SIZE_WEIGHTS)[0]
    if rng.random() < 0.95:
        el["stock"] = rng.choices([0, 1, 2, 3, 5, 8, 12, 20, 50],
                                  [25, 12, 12, 12, 12, 10, 8, 5, 4])[0]
    if rng.random() < 0.98:
        base = rng.choice([39, 59, 79, 99, 129, 159, 199, 259, 399])
        el["price"] = base + rng.choice([0, 0, 0, 9, 90])
    if rng.random() < 0.30:
        el["sku"] = "SKU-%06d" % rng.randrange(1_000_000)
    if rng.random() < 0.10:
        el["tags"] = rng.sample(["new", "sale", "hot", "limited"], rng.randint(1, 3))
    if weird:
        r = rng.random()
        if r < 0.30:
            el["price"] = rng.choice(["99.00", "N/A", None])
        elif r < 0.55:
            el["stock"] = rng.choice([True, False, 3.0, 0.5, "5"])
        elif r < 0.75:
            el["size"] = rng.choice([1, 0, None])
        elif r < 0.90:
            el["color"] = rng.choice([None, 7, "Red "])
        # else: leave a normal-looking element
    return el


def make_product(doc_id: int, rng: random.Random) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"_id": "P%07d" % doc_id}
    doc["name"] = f"{rng.choice(CATEGORIES).title()} {doc_id}"
    doc["brand"] = rng.choice(BRANDS)
    doc["season"] = rng.choice(SEASONS)
    doc["category"] = rng.choice(CATEGORIES)
    doc["rating"] = round(rng.uniform(1.0, 5.0), 1)
    doc["updated_at"] = "2025-%02d-%02d" % (rng.randint(1, 12), rng.randint(1, 28))

    roll = rng.random()
    if roll < 0.02:
        # missing specs entirely
        return doc
    if roll < 0.05:
        doc["specs"] = []
        return doc
    if roll < 0.06:
        # mixed type: specs is not an array
        doc["specs"] = rng.choice(["no-specs", 0, {"color": "red", "size": "XL"}])
        return doc
    if roll < 0.07:
        # trap documents: colour and size split across elements
        doc["specs"] = [
            spec_element(rng, weird=True),
            {"color": rng.choice(COLORS), "size": rng.choice(SIZES),
             "stock": rng.choice([0, 0, 5]), "price": rng.choice([99, 199])},
        ]
        doc["specs"][0]["color"] = "red"
        doc["specs"][0]["size"] = "S"
        doc["specs"][1]["color"] = rng.choice(["blue", "green", "black"])
        doc["specs"][1]["size"] = "XL"
        return doc

    n = rng.choices([0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [0, 20, 25, 20, 14, 8, 4, 3, 2])[0]
    if n == 0:
        doc["specs"] = []
        return doc
    doc["specs"] = [spec_element(rng, weird=(rng.random() < 0.12)) for _ in range(n)]
    return doc


def generate(count: int, seed: int = 42) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    for i in range(1, count + 1):
        yield make_product(i, rng)
