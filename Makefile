.PHONY: test bench serve clean

COUNT ?= 300000
SEED ?= 42
DATA ?= data/catalog
PORT ?= 8765

test:
	python3 -m pytest -q

bench:
	python3 -m docstore.bench --count $(COUNT) --seed $(SEED) --out $(DATA)

serve:
	python3 -m docstore.server --data $(DATA) --port $(PORT)

clean:
	rm -rf $(DATA) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
