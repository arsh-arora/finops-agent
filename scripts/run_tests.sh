#!/bin/bash

# Test runner script for FinOps Agent Chat

set -e

echo "🧪 Running FinOps Agent Chat Test Suite"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements/base.txt
pip install -r requirements/test.txt

# Create test database directory if needed
mkdir -p tests/data

# Run different test categories
echo ""
echo "🔧 Running Unit Tests..."
echo "------------------------"
pytest tests/unit/ -v --tb=short -m "unit"

echo ""
echo "🔗 Running Integration Tests..."
echo "-------------------------------"
pytest tests/integration/ -v --tb=short -m "integration"

echo ""
echo "📊 Running Full Test Suite with Coverage..."
echo "--------------------------------------------"
pytest --cov=src --cov-report=term-missing --cov-report=html:htmlcov --tb=short

echo ""
echo "✅ All tests completed!"
echo ""
echo "📈 Coverage report available in htmlcov/index.html"
echo "🔍 Test results summary:"
pytest --tb=no -q

# Check if coverage is above threshold
echo ""
echo "📊 Coverage Summary:"
coverage report --show-missing --skip-covered