.PHONY: help venv install lint typecheck test check up down fixtures

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

venv:            ## create the virtualenv
	python3 -m venv $(VENV)

install: venv    ## install the package and dev dependencies
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

lint:            ## ruff check + format check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

typecheck:       ## mypy strict
	$(VENV)/bin/mypy

test:            ## pytest with coverage gate
	$(VENV)/bin/pytest

check: lint typecheck test  ## everything CI runs

up:              ## start postgres + neo4j
	docker compose up -d

down:            ## stop databases
	docker compose down

fixtures:        ## refetch the committed wikitext fixtures from the live wiki
	$(VENV)/bin/python scripts/fetch_fixtures.py
