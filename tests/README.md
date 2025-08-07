# FinOps Agent Chat - Test Suite

This directory contains comprehensive tests for Phase 1 of the FinOps Agent Chat system.

## Test Structure

```
tests/
├── conftest.py              # Global fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── database/           # Database layer tests
│   │   ├── test_connection.py
│   │   ├── test_models.py
│   │   ├── test_redis_client.py
│   │   ├── test_qdrant_client.py
│   │   └── test_neo4j_client.py
│   ├── auth/               # Authentication tests
│   │   └── test_jwt_auth.py
│   └── websocket/          # WebSocket layer tests
│       ├── test_connection_manager.py
│       └── test_handler.py
└── integration/            # Integration tests (slower, cross-component)
    ├── test_fastapi_app.py
    ├── test_websocket_flow.py
    └── test_database_integration.py
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Database Layer Tests:**
- `test_connection.py` - Database connection and session management
- `test_models.py` - ConversationEvent model and enum validation
- `test_redis_client.py` - Redis operations (set, get, publish, etc.)
- `test_qdrant_client.py` - Vector database operations
- `test_neo4j_client.py` - Graph database operations

**Authentication Tests:**
- `test_jwt_auth.py` - JWT creation, validation, and user extraction

**WebSocket Tests:**
- `test_connection_manager.py` - Connection lifecycle, user tracking, messaging
- `test_handler.py` - Message processing and event creation

### Integration Tests (`tests/integration/`)

**FastAPI Application:**
- Complete application startup and shutdown
- Health check endpoints
- Authentication token creation
- WebSocket endpoint integration

**WebSocket Flow:**
- End-to-end connection establishment
- Complete message handling flow
- Multi-user conversation scenarios
- Error handling and cleanup

**Database Integration:**
- Cross-database operations
- Complex queries and data relationships
- Performance and concurrency testing

## Running Tests

### Quick Development Testing
```bash
# Run only fast unit tests
./scripts/test_quick.sh

# Or manually
pytest tests/unit/ -v --tb=short -x --ff
```

### Full Test Suite
```bash
# Run all tests with coverage
./scripts/run_tests.sh

# Or manually
pytest --cov=src --cov-report=term-missing --cov-report=html
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v -m unit

# Integration tests only  
pytest tests/integration/ -v -m integration

# Database tests only
pytest -v -m database

# WebSocket tests only
pytest -v -m websocket

# Authentication tests only
pytest -v -m auth

# Slow tests (for CI/CD)
pytest -v -m slow
```

### Test Filtering
```bash
# Run specific test file
pytest tests/unit/database/test_models.py -v

# Run specific test method
pytest tests/unit/database/test_models.py::TestConversationEventModel::test_conversation_event_creation -v

# Run tests matching pattern
pytest -k "test_websocket" -v

# Skip slow tests
pytest -v -m "not slow"
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Coverage reporting enabled
- Custom markers for test categorization
- Async mode configured
- Warning filters applied

### Test Fixtures (`conftest.py`)
- Database sessions with in-memory SQLite
- Mock clients for Redis, Qdrant, Neo4j
- JWT authentication helpers
- Sample data factories
- WebSocket mocks

## Test Data and Mocking

### Mock Strategy
- **Database clients**: Mocked for unit tests, real in-memory/test DBs for integration
- **WebSocket connections**: AsyncMock objects with send/receive simulation
- **External services**: Fully mocked to ensure test isolation

### Test Data
- **Factories**: ConversationEventFactory for creating test models
- **Fixtures**: Pre-configured sample data (users, messages, tokens)
- **UUID generation**: Controlled for predictable test outcomes

## Coverage Goals

### Current Coverage Targets
- **Overall**: >90%
- **Critical paths**: 100% (authentication, message handling, database operations)
- **Error handling**: >95%
- **Integration flows**: >85%

### Coverage Reports
```bash
# View coverage in terminal
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Continuous Integration

### GitHub Actions Integration
```yaml
# Example CI configuration
- name: Run Tests
  run: |
    pip install -r requirements/base.txt
    pip install -r requirements/test.txt
    pytest --cov=src --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Debugging Failed Tests

### Common Issues and Solutions

**Database Connection Errors:**
```bash
# Check database setup
pytest tests/unit/database/test_connection.py::test_check_database_connection_success -v -s
```

**WebSocket Mock Issues:**
```bash
# Run with debug output
pytest tests/unit/websocket/ -v -s --tb=long
```

**Async Test Problems:**
```bash
# Ensure async mode is configured
pytest tests/integration/ -v --tb=long -s
```

### Test Debugging
```bash
# Run single test with full output
pytest tests/path/to/test.py::TestClass::test_method -v -s --tb=long

# Drop into debugger on failure
pytest --pdb tests/path/to/test.py

# Show local variables in traceback
pytest --tb=long --showlocals tests/path/to/test.py
```

## Best Practices

### Writing Tests
1. **AAA Pattern**: Arrange, Act, Assert
2. **Descriptive names**: `test_websocket_connection_with_invalid_token_returns_401`
3. **One concept per test**: Test one specific behavior
4. **Mock external dependencies**: Keep tests isolated and fast
5. **Use fixtures**: Reuse common setup code

### Test Organization
1. **Mirror source structure**: `src/database/models.py` → `tests/unit/database/test_models.py`
2. **Group related tests**: Use test classes for logical grouping
3. **Mark appropriately**: Use pytest markers (@pytest.mark.unit, etc.)
4. **Separate concerns**: Unit vs integration vs end-to-end

### Performance
1. **Fast feedback**: Unit tests should run in <10 seconds
2. **Parallel execution**: Use pytest-xdist for CI/CD
3. **Test data cleanup**: Use fixtures for automatic cleanup
4. **Mock expensive operations**: Database calls, network requests

## Adding New Tests

### For New Features
1. **Start with unit tests**: Test individual components
2. **Add integration tests**: Test component interactions
3. **Update fixtures**: Add necessary test data
4. **Update documentation**: Keep this README current

### Test Template
```python
"""
Tests for new component.
"""

import pytest
from src.new_component import NewComponent


@pytest.mark.unit
class TestNewComponent:
    """Test NewComponent functionality."""
    
    def test_new_component_creation(self):
        """Test creating NewComponent instance."""
        component = NewComponent()
        assert component is not None
    
    async def test_async_operation(self):
        """Test async operation."""
        component = NewComponent()
        result = await component.async_method()
        assert result == "expected"
```

## Troubleshooting

### Common Test Failures

**Import Errors**: Check PYTHONPATH and package structure
**Async Errors**: Ensure pytest-asyncio is installed and configured
**Database Errors**: Verify test database setup in conftest.py
**Mock Errors**: Check mock setup and return values

### Getting Help

1. Check test output and error messages carefully
2. Run with increased verbosity: `-v -s --tb=long`
3. Check similar existing tests for patterns
4. Review fixture setup in `conftest.py`
5. Consult pytest documentation for advanced features