"""
End-to-End Tests for FinOps Agent Chat System

This package contains comprehensive end-to-end tests that verify the complete
system integration from WebSocket connections through agent execution and 
memory persistence.

Test Structure:
- test_ws_streaming.py: WebSocket streaming and lifecycle events
- test_multi_agent_orchestration.py: Multi-agent workflow coordination
- test_memory_roundtrip.py: Memory persistence and retrieval verification
- utils/: Test utilities and WebSocket client
- fixtures/: Test data and sample files

Requirements:
- Real database instances (PostgreSQL, Redis, Neo4j, Qdrant)
- Celery workers for distributed task testing
- Agent system with memory service integration
"""