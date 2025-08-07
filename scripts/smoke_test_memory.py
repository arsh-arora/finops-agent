#!/usr/bin/env python3
"""
Smoke test for FinOps Memory System with Mem0.

This script validates basic functionality of the memory system
including initialization, storage, retrieval, and health checks.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import FinOpsMemoryService, ConversationContext, FinOpsMemoryCategory, MemoryPriority
from memory.exceptions import MemoryServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_memory_initialization():
    """Test memory service initialization."""
    logger.info("Testing memory service initialization...")
    
    service = FinOpsMemoryService()
    
    # Test health status before initialization
    status = service.get_health_status()
    logger.info(f"Pre-init health status: {status}")
    
    try:
        # Initialize memory service
        success = await service.initialize()
        
        if success:
            logger.info("✅ Memory service initialized successfully")
            
            # Test health status after initialization  
            status = service.get_health_status()
            logger.info(f"Post-init health status: {status}")
            
            return service
        else:
            logger.error("❌ Memory service initialization failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Memory service initialization error: {e}")
        return None


async def test_memory_storage_and_retrieval(service: FinOpsMemoryService):
    """Test memory storage and retrieval functionality."""
    logger.info("Testing memory storage and retrieval...")
    
    try:
        # Create conversation context
        context = ConversationContext(
            user_id="smoke_test_user",
            conversation_id="smoke_test_conv",
            category=FinOpsMemoryCategory.COST_ANALYSIS,
            priority=MemoryPriority.HIGH,
            metadata={"test_run": True}
        )
        
        # Test memory storage
        test_messages = [
            {"role": "user", "content": "What are my AWS costs for this month?"},
            {"role": "assistant", "content": "Let me analyze your AWS costs. Your total spending this month is $2,450, which is 15% higher than last month."}
        ]
        
        logger.info("Storing test conversation...")
        memory_id = await service.store_conversation_memory(test_messages, context)
        logger.info(f"✅ Stored memory with ID: {memory_id}")
        
        # Test memory retrieval
        logger.info("Retrieving relevant memories...")
        memories = await service.retrieve_relevant_memories(
            "AWS costs analysis",
            context
        )
        logger.info(f"✅ Retrieved {len(memories)} relevant memories")
        
        if memories:
            first_memory = memories[0]
            content = ""
            if isinstance(first_memory, dict):
                content = first_memory.get('content', first_memory.get('memory', ''))
            else:
                content = str(first_memory)
            logger.info(f"First memory: {content[:100]}...")
        
        # Test user memories
        logger.info("Getting all user memories...")
        user_memories = await service.get_user_memories("smoke_test_user")
        logger.info(f"✅ Retrieved {len(user_memories)} user memories")
        
        # Test memory statistics
        logger.info("Getting memory statistics...")
        stats = await service.get_memory_stats("smoke_test_user")
        logger.info(f"✅ Memory stats: {stats.total_memories} total memories")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory storage/retrieval test failed: {e}")
        return False


async def test_memory_cleanup(service: FinOpsMemoryService):
    """Clean up test memories."""
    logger.info("Cleaning up test memories...")
    
    try:
        # Get all user memories for cleanup
        user_memories = await service.get_user_memories("smoke_test_user")
        
        deleted_count = 0
        for memory in user_memories:
            memory_id = None
            
            # Handle different memory formats with better extraction
            if isinstance(memory, dict):
                memory_id = memory.get("id") or memory.get("memory_id")
            elif isinstance(memory, str):
                # For string memories in fallback mode, skip individual deletion
                logger.debug(f"Skipping string-format memory: {memory[:30]}...")
                continue
            else:
                # Try to get ID from object attributes
                memory_id = getattr(memory, 'id', None) or getattr(memory, 'memory_id', None)
            
            # Only try to delete if we have a valid UUID-format memory_id
            if memory_id and memory_id != "unknown" and not memory_id.startswith("string_mem") and not memory_id.startswith("fallback_mem"):
                try:
                    success = await service.delete_memory(memory_id)
                    if success:
                        deleted_count += 1
                        logger.info(f"Deleted memory: {memory_id}")
                except Exception as delete_error:
                    logger.warning(f"Failed to delete memory {memory_id}: {delete_error}")
            elif memory_id:
                logger.debug(f"Skipping invalid memory ID format: {memory_id}")
        
        # Try to clean up any remaining memories by user_id
        try:
            # Use Mem0's delete_all for complete cleanup
            service.memory.delete_all(user_id="smoke_test_user")
            logger.info(f"✅ Cleaned up {deleted_count} individual memories + batch delete for user")
        except Exception as batch_delete_error:
            logger.warning(f"Batch delete failed: {batch_delete_error}")
            logger.info(f"✅ Cleaned up {deleted_count} individual memories")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory cleanup failed: {e}")
        return False


async def run_smoke_tests():
    """Run all smoke tests for the memory system."""
    logger.info("🚀 Starting FinOps Memory System smoke tests...")
    
    # Check required environment variables (Groq API key is set in config)
    # The Groq API key is automatically set in the configuration
    logger.info("✅ Using Groq API + Hugging Face embeddings (completely free)")
    
    test_results = []
    
    # Test 1: Memory Service Initialization
    service = await test_memory_initialization()
    test_results.append(service is not None)
    
    if not service:
        logger.error("❌ Cannot continue tests without initialized service")
        return False
    
    # Test 2: Memory Storage and Retrieval
    storage_success = await test_memory_storage_and_retrieval(service)
    test_results.append(storage_success)
    
    # Test 3: Memory Cleanup
    cleanup_success = await test_memory_cleanup(service)
    test_results.append(cleanup_success)
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info(f"\n📊 Smoke Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All smoke tests passed! Memory system is functional.")
        return True
    else:
        logger.error("❌ Some smoke tests failed. Check logs for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_smoke_tests())
    sys.exit(0 if success else 1)