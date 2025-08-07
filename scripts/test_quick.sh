#!/bin/bash

# Quick test runner for development

set -e

echo "⚡ Running Quick Tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run only unit tests (fast)
pytest tests/unit/ -v --tb=short -x --ff

echo "✅ Quick tests completed!"