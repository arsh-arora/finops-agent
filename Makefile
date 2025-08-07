# FinOps Agent Chat - Development Makefile

.PHONY: help install test test_unit test_integration test_quick test_coverage clean lint format docker_build docker_up docker_down

# Default target
help: ## Show this help message
	@echo "FinOps Agent Chat - Development Commands"
	@echo "======================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install all dependencies
	pip install -r requirements/base.txt
	pip install -r requirements/test.txt

install_dev: ## Install development dependencies
	pip install -r requirements/base.txt
	pip install -r requirements/test.txt
	pip install pre-commit black flake8 isort
	pre-commit install

# Testing
test: ## Run full test suite with coverage
	@echo "🧪 Running full test suite..."
	python -m pytest --cov=src --cov-report=term-missing --cov-report=html:htmlcov

test_unit: ## Run unit tests only
	@echo "⚡ Running unit tests..."
	python -m pytest tests/unit/ -v -m unit

test_integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	python -m pytest tests/integration/ -v -m integration

test_quick: ## Run quick tests (unit tests, fail fast)
	@echo "🚀 Running quick tests..."
	python -m pytest tests/unit/ -v --tb=short -x --ff

test_coverage: ## Generate detailed coverage report
	@echo "📊 Generating coverage report..."
	python -m pytest --cov=src --cov-report=html:htmlcov --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test_watch: ## Run tests continuously (requires pytest-watch)
	@echo "👀 Running tests in watch mode..."
	ptw tests/ src/

# Code Quality
lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/ || echo "flake8 not installed, skipping..."
	black --check src/ tests/ || echo "black not installed, skipping..."
	isort --check-only src/ tests/ || echo "isort not installed, skipping..."

format: ## Format code with black and isort
	@echo "✨ Formatting code..."
	black src/ tests/ || echo "black not installed, skipping..."
	isort src/ tests/ || echo "isort not installed, skipping..."

format_check: ## Check code formatting without making changes
	@echo "🔍 Checking code formatting..."
	black --check --diff src/ tests/ || echo "black not installed"
	isort --check-only --diff src/ tests/ || echo "isort not installed"

# Database
db_init: ## Initialize all databases
	@echo "🗄️ Initializing databases..."
	python -m src.database.init_db

db_health: ## Check database health
	@echo "🏥 Checking database health..."
	python -c "import asyncio; from src.database.init_db import health_check; print(asyncio.run(health_check()))"

# Docker
docker_build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t finops-agent-chat .

docker_up: ## Start all services with docker-compose
	@echo "🚀 Starting services..."
	docker-compose up -d

docker_down: ## Stop all services
	@echo "🛑 Stopping services..."
	docker-compose down

docker_logs: ## View logs from all services
	@echo "📜 Viewing logs..."
	docker-compose logs -f

docker_test: ## Run tests in Docker
	@echo "🧪 Running tests in Docker..."
	docker-compose run --rm app python -m pytest

# Application
run: ## Run the application locally
	@echo "🚀 Starting FinOps Agent Chat..."
	python main.py

run_dev: ## Run in development mode with auto-reload
	@echo "🔧 Starting in development mode..."
	DEBUG=true uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Cleanup
clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/

clean_db: ## Clean up test databases
	@echo "🗑️ Cleaning test databases..."
	rm -rf tests/data/
	mkdir -p tests/data

# Development Utilities
check: ## Run all checks (tests, lint, format)
	@echo "✅ Running all checks..."
	make test_unit
	make lint
	make format_check

fix: ## Fix common issues (format, sort imports)
	@echo "🔧 Fixing common issues..."
	make format

# CI/CD
ci: ## Run CI pipeline locally
	@echo "🤖 Running CI pipeline..."
	make clean
	make install
	make lint
	make test_coverage
	@echo "✅ CI pipeline completed successfully!"

# Example Usage
example: ## Show example usage
	@echo "📖 Example Usage:"
	@echo ""
	@echo "1. Setup development environment:"
	@echo "   make install"
	@echo ""
	@echo "2. Start databases:"
	@echo "   make docker_up"
	@echo ""
	@echo "3. Run tests:"
	@echo "   make test_quick"
	@echo ""
	@echo "4. Start application:"
	@echo "   make run_dev"
	@echo ""
	@echo "5. Access application:"
	@echo "   http://localhost:8000"
	@echo "   http://localhost:8000/docs (API docs)"