.PHONY: sync validate train test web build

sync:
	uv sync

validate:
	for i in 01 02 03 04 05; do uv run python python/validate_level.py shared/levels/level$$i.json || exit 1; done

train:
	for i in 01 02 03 04 05; do \
		uv run python python/train.py --level shared/levels/level$$i.json --mode bellman || exit 1; \
		uv run python python/train.py --level shared/levels/level$$i.json --mode qlearning --episodes 4000 --seed 1 || exit 1; \
	done

test:
	uv run black --check python
	uv run flake8 python
	uv run pytest -q

web:
	cd web && npm run dev

build:
	cd web && npm run build
