#!/usr/bin/env python3
"""
Integrated Demo for FinOps Multi-Agent System
Uses the actual system components, WebSocket bridge, and real agent execution
"""

import asyncio
import json
import time
from typing import Dict, Any
from datetime import datetime, timezone

# Import the real system components
from config.settings import settings
from src.database.init_db import initialize_all_databases, cleanup_connections
from src.websocket.agent_integration import websocket_agent_bridge
from src.websocket.connection_manager import ConnectionManager
from src.agents.models import ChatRequest
from src.agents import initialize_agent_registry
from src.memory.mem0_service import FinOpsMemoryService
from src.memory.config import get_mem0_config
from src.llm.openrouter_client import OpenRouterClient
from src.database.connection import get_db_session
from src.auth import create_access_token


class IntegratedSystemDemo:
    """Demo that uses the actual integrated system components"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.test_scenarios = {
            "finops": [
                "Analyze my AWS costs for the last month and identify optimization opportunities",
                "Calculate NPV for a $100K cloud migration project with 20% cost savings over 3 years",
                "Detect any cost anomalies in my spending patterns using machine learning",
                "Optimize budget allocation across development, staging, and production environments"
            ],
            "github": [
                "Perform comprehensive security analysis of repository: https://github.com/example/vulnerable-app",
                "Check for dependency vulnerabilities and provide EPSS risk scores", 
                "Analyze contributor behavior patterns for anomalies and security risks",
                "Compare security posture across multiple repositories"
            ],
            "document": [
                "Extract financial tables from this quarterly report PDF with bounding box coordinates",
                "Analyze document content with ML-driven insights and entity extraction",
                "Process multi-format documents and extract key information",
                "Compare multiple documents for similarities and differences"
            ],
            "research": [
                "Research latest trends in cloud cost optimization and FinOps for 2024",
                "Find and verify information about multi-cloud security best practices",
                "Fact-check recent developments in container orchestration costs",
                "Research market consensus on cloud infrastructure pricing trends"
            ],
            "deep_research": [
                "Coordinate comprehensive analysis of cloud costs, security vulnerabilities, and market trends",
                "Orchestrate multi-agent research on FinOps best practices across financial and security domains",
                "Synthesize insights from document analysis, web research, and financial modeling",
                "Perform adaptive research with cross-domain validation on cloud optimization"
            ]
        }
    
    async def initialize_system(self):
        """Initialize the complete integrated system"""
        print("🚀 Initializing Complete FinOps Multi-Agent System")
        print("=" * 60)
        
        try:
            # Initialize all databases
            print("🗄️  Initializing databases...")
            db_success = await initialize_all_databases()
            if not db_success:
                print("❌ Database initialization failed")
                return False
            print("✅ Databases initialized successfully")
            
            # Initialize memory service
            print("🧠 Initializing memory service...")
            memory_config = get_mem0_config()
            memory_service = FinOpsMemoryService(config=memory_config)
            await memory_service.initialize()
            print("✅ Memory service initialized")
            
            # Initialize OpenRouter LLM client
            print("🤖 Initializing OpenRouter LLM client...")
            llm_client = OpenRouterClient()
            print("✅ OpenRouter client initialized")
            
            # Initialize agent registry with memory service and OpenRouter
            print("🤖 Initializing agent registry...")
            agent_registry = initialize_agent_registry(
                memory_service=memory_service,
                llm_client=llm_client,
                config={}
            )
            print("✅ Agent registry initialized")
            
            # Initialize WebSocket-Agent bridge with connection manager
            print("🌉 Initializing WebSocket-Agent bridge...")
            websocket_agent_bridge.connection_manager = self.connection_manager
            await websocket_agent_bridge.initialize()
            print("✅ WebSocket-Agent bridge initialized")
            
            print("\n🎉 Complete system initialization successful!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def simulate_websocket_connection(self, user_id: str, conversation_id: str) -> str:
        """Simulate a WebSocket connection and return connection ID"""
        
        # Create JWT token for the user
        token_data = {"sub": user_id, "username": f"demo_{user_id}"}
        access_token = create_access_token(data=token_data)
        
        # Simulate WebSocket connection
        connection_id = f"conn_{user_id}_{int(time.time())}"
        
        # Add connection to manager (simulating WebSocket auth)
        connection_info = {
            'user_id': user_id,
            'conversation_id': conversation_id,
            'token': access_token,
            'connected_at': datetime.now(timezone.utc),
            'websocket': None  # Would be actual WebSocket in real scenario
        }
        
        # Store connection info
        self.connection_manager.active_connections[connection_id] = connection_info
        
        print(f"🔌 Simulated WebSocket connection: {connection_id}")
        return connection_id
    
    async def test_complete_agent_pipeline(self):
        """Test the complete agent pipeline with real components"""
        print("\n🧪 Testing Complete Agent Pipeline")
        print("=" * 60)
        
        for domain, scenarios in self.test_scenarios.items():
            print(f"\n🎯 Testing {domain.upper()} Domain:")
            print("-" * 40)
            
            # Test first scenario for each domain
            scenario = scenarios[0]
            user_id = f"demo_user_{domain}"
            conversation_id = f"conv_{domain}_{int(time.time())}"
            
            print(f"📝 Scenario: {scenario}")
            
            try:
                # Simulate WebSocket connection
                connection_id = await self.simulate_websocket_connection(user_id, conversation_id)
                
                # Create message in WebSocket format
                message = {
                    'type': 'user_message',
                    'text': scenario,
                    'metadata': {
                        'source': 'integrated_demo',
                        'domain_test': domain,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message_id': f"msg_{domain}_{int(time.time())}"
                }
                
                print(f"   🔌 Connection ID: {connection_id}")
                print(f"   👤 User ID: {user_id}")
                print(f"   💬 Conversation ID: {conversation_id}")
                
                # Collect execution events
                execution_events = []
                
                # Mock sending messages back to collect events
                original_send = self.connection_manager.send_personal_message
                
                async def capture_send(connection_id: str, message: Dict[str, Any]):
                    execution_events.append(message)
                    print(f"   📡 Event: {message.get('event', 'message')} - {message.get('message', '')}")
                
                self.connection_manager.send_personal_message = capture_send
                
                # Process message through the complete pipeline
                start_time = time.time()
                
                # Create a mock database session (in real system this comes from FastAPI dependency)
                from src.database.connection import get_db_session
                async with get_db_session() as db_session:
                    await websocket_agent_bridge.process_user_message(
                        connection_id=connection_id,
                        message=message,
                        db_session=db_session
                    )
                
                # Wait for execution to complete
                await asyncio.sleep(2.0)  # Give time for async execution
                
                processing_time = (time.time() - start_time) * 1000
                
                # Restore original send method
                self.connection_manager.send_personal_message = original_send
                
                # Analyze results
                print(f"   ⏱️  Total processing time: {processing_time:.2f}ms")
                print(f"   📊 Execution events captured: {len(execution_events)}")
                
                # Show key events
                key_events = ['execution_started', 'agent_selected', 'plan_created', 'graph_compiled', 'execution_completed']
                for event_name in key_events:
                    event = next((e for e in execution_events if e.get('event') == event_name), None)
                    if event:
                        if event_name == 'agent_selected':
                            print(f"   ✅ Agent: {event.get('agent_domain', 'unknown')}")
                        elif event_name == 'plan_created':
                            print(f"   ✅ Plan: {event.get('task_count', 0)} tasks")
                        elif event_name == 'graph_compiled':
                            print(f"   ✅ Graph: {event.get('node_count', 0)} nodes, {event.get('edge_count', 0)} edges")
                        else:
                            print(f"   ✅ {event_name.replace('_', ' ').title()}")
                    else:
                        print(f"   ⏳ {event_name.replace('_', ' ').title()}: In Progress")
                
                # Clean up connection
                if connection_id in self.connection_manager.active_connections:
                    del self.connection_manager.active_connections[connection_id]
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print()
            await asyncio.sleep(1.0)  # Pause between domains
    
    async def test_cross_domain_coordination(self):
        """Test cross-domain agent coordination"""
        print("\n🌐 Testing Cross-Domain Agent Coordination")
        print("=" * 60)
        
        complex_scenario = """
        I need a comprehensive business analysis:
        
        1. Analyze our cloud spending trends and optimization opportunities
        2. Check our main repositories for security vulnerabilities
        3. Extract key metrics from our Q4 financial reports
        4. Research industry benchmarks for our sector
        5. Coordinate all findings into an executive summary
        
        Please orchestrate this multi-domain analysis with cross-validation.
        """
        
        print("📋 Multi-Domain Scenario:")
        print(complex_scenario)
        
        try:
            user_id = "demo_executive"
            conversation_id = f"exec_analysis_{int(time.time())}"
            connection_id = await self.simulate_websocket_connection(user_id, conversation_id)
            
            message = {
                'type': 'user_message',
                'text': complex_scenario,
                'metadata': {
                    'source': 'integrated_demo',
                    'complexity': 'high',
                    'cross_domain': True
                },
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message_id': f"msg_cross_domain_{int(time.time())}"
            }
            
            print(f"🔌 Connection: {connection_id}")
            print("🚀 Processing multi-domain request...")
            
            # Track execution events
            execution_events = []
            
            async def capture_events(connection_id: str, message: Dict[str, Any]):
                execution_events.append(message)
                event_type = message.get('event', 'message')
                print(f"   📡 {event_type}: {message.get('message', message.get('agent_domain', ''))}")
            
            self.connection_manager.send_personal_message = capture_events
            
            # Process through pipeline
            async with get_db_session() as db_session:
                await websocket_agent_bridge.process_user_message(
                    connection_id=connection_id,
                    message=message,
                    db_session=db_session
                )
            
            # Wait for completion
            await asyncio.sleep(3.0)
            
            print(f"\n📊 Cross-domain execution completed with {len(execution_events)} events")
            
            # Show coordination result
            agent_selected = next((e for e in execution_events if e.get('event') == 'agent_selected'), None)
            if agent_selected:
                selected_agent = agent_selected.get('agent_domain', 'unknown')
                print(f"🎯 Primary coordinator: {selected_agent}")
                
                if selected_agent == 'deep_research':
                    print("✅ Correctly routed to Deep Research Agent for multi-domain coordination")
                else:
                    print(f"ℹ️  Routed to {selected_agent} - single domain focus")
            
            # Clean up
            if connection_id in self.connection_manager.active_connections:
                del self.connection_manager.active_connections[connection_id]
            
        except Exception as e:
            print(f"❌ Cross-domain test failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def show_system_status(self):
        """Show integrated system status"""
        print("\n📊 Integrated System Status")
        print("=" * 60)
        
        try:
            # WebSocket bridge status
            active_executions = websocket_agent_bridge.get_active_executions()
            print(f"🔄 Active executions: {len(active_executions)}")
            
            # Connection manager status  
            active_connections = len(self.connection_manager.active_connections)
            print(f"🔌 Active connections: {active_connections}")
            
            # Agent registry health
            if websocket_agent_bridge.agent_registry:
                health = await websocket_agent_bridge.agent_registry.health_check()
                print(f"🏥 Agent registry: {health['registry_status']}")
                print(f"🤖 Available agents: {health['total_agents']}")
                print(f"🧠 Memory service: {health['memory_service_status']}")
                
                # Routing statistics
                routing_stats = websocket_agent_bridge.agent_registry.get_routing_stats()
                print(f"🎯 Routing success rate: {routing_stats['success_rate_percent']:.1f}%")
            
            # Memory service status
            if websocket_agent_bridge.memory_service:
                print("🧠 Memory service: Connected and operational")
            
        except Exception as e:
            print(f"❌ Status check error: {e}")
    
    async def cleanup_system(self):
        """Clean up the integrated system"""
        print("\n🧹 Cleaning up integrated system...")
        
        try:
            # Clean up WebSocket bridge
            await websocket_agent_bridge.cleanup()
            print("✅ WebSocket bridge cleaned up")
            
            # Clean up database connections
            await cleanup_connections()
            print("✅ Database connections closed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    async def run_complete_demo(self):
        """Run the complete integrated system demo"""
        print("🎉 FinOps Multi-Agent System - INTEGRATED DEMO")
        print("=" * 70)
        print("Testing the complete production system with real components")
        print()
        
        try:
            # Initialize the complete system
            if not await self.initialize_system():
                print("❌ Failed to initialize system - exiting")
                return
            
            # Test complete agent pipeline
            await self.test_complete_agent_pipeline()
            
            # Test cross-domain coordination
            await self.test_cross_domain_coordination()
            
            # Show system status
            await self.show_system_status()
            
            print("\n🎉 Integrated Demo Completed Successfully!")
            print("\n💡 System is ready for:")
            print("   • Full WebSocket connections via main.py")
            print("   • Web interface via demo.html")  
            print("   • Production deployment via docker-compose")
            print("   • API access via FastAPI endpoints")
            
        except Exception as e:
            print(f"\n❌ Integrated demo failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            await self.cleanup_system()


async def main():
    """Main entry point for integrated demo"""
    demo = IntegratedSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("🔧 Note: This demo requires database services to be running")
    print("   Start with: docker-compose up -d postgres redis neo4j qdrant")
    print("   Or set environment variables for external services")
    print()
    
    # Run the integrated demo
    asyncio.run(main())