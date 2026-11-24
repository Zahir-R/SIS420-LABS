.PHONY: sync validate train test web build

sync:
	uv sync

validate:
	uv run python python/validate_level.py shared/levels/level01.json

train:
	uv run python python/train.py --mode bellman
	uv run python python/train.py --mode qlearning --episodes 4000 --seed 1

test:
	uv run black --check python
	uv run flake8 python
	uv run pytest -q

web:
	cd web && npm run dev

build:
	cd web && npm run build
