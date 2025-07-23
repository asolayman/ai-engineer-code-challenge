.PHONY: help install test lint format clean docs serve-docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .[dev]

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/

docs: ## Build documentation
	cd docs && make html

serve-docs: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

security: ## Run security checks
	bandit -r src/
	safety check

ci: ## Run CI checks locally
	make lint
	make test-cov
	make security
	make docs

ingest: ## Ingest sample documents
	python main.py --mode ingest --documents ./data/

query: ## Run a sample query
	python main.py --mode query --query "What are the key features of the AI Code Challenge?" --verbose

all: ## Run all checks
	make format
	make lint
	make test-cov
	make security
	make docs 
