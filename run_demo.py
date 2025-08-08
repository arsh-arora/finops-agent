#!/usr/bin/env python3
"""
Demo Runner for FinOps Multi-Agent System

This script provides different ways to test and demonstrate the system:
1. Quick demo with the real integrated system
2. Full web interface demo  
3. WebSocket client demo
4. Production server startup
"""

import asyncio
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path


def run_integrated_demo():
    """Run the integrated system demo"""
    print("🚀 Running Integrated System Demo")
    print("This will test the real agent system with all components")
    print()
    
    try:
        subprocess.run(["python", "demo_integrated.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code {e.returncode}")
    except FileNotFoundError:
        print("❌ Python not found. Make sure Python is installed and in PATH")


def run_web_demo():
    """Run the web interface demo"""
    print("🌐 Starting Web Interface Demo")
    print("This will start the server and open the web demo")
    print()
    
    # Start the server in background
    server_process = None
    try:
        print("🚀 Starting FastAPI server...")
        server_process = subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✅ Server started successfully")
            
            # Open web demo
            demo_path = Path("demo.html").absolute()
            if demo_path.exists():
                print("🌐 Opening web demo...")
                webbrowser.open(f"file://{demo_path}")
                print("💡 WebSocket endpoint: ws://localhost:8000/api/v1/ws/chat")
                print("💡 API docs: http://localhost:8000/docs")
                print("\nPress Ctrl+C to stop the server")
                
                # Keep running until interrupted
                try:
                    server_process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
                    server_process.terminate()
                    server_process.wait()
            else:
                print("❌ demo.html not found")
                server_process.terminate()
        else:
            print("❌ Failed to start server")
            if server_process.stdout:
                print("Server output:", server_process.stdout.read().decode())
            if server_process.stderr:
                print("Server errors:", server_process.stderr.read().decode())
                
    except FileNotFoundError:
        print("❌ Python not found. Make sure Python is installed and in PATH")
    except Exception as e:
        print(f"❌ Error running web demo: {e}")
    finally:
        if server_process and server_process.poll() is None:
            server_process.terminate()


def run_websocket_client():
    """Run a simple WebSocket client demo"""
    print("🔌 Running WebSocket Client Demo")
    print("This will connect to the WebSocket API and test messaging")
    print()
    
    websocket_client_code = '''
import asyncio
import websockets
import json
from datetime import datetime

async def websocket_demo():
    uri = "ws://localhost:8000/api/v1/ws/chat"
    
    # You'll need to get a JWT token first
    # In production, this would come from authentication
    token = "demo_token"  # Replace with real token from /auth/token
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(f"{uri}?token={token}") as websocket:
            print("✅ Connected!")
            
            # Send test message
            test_message = {
                "type": "user_message",
                "text": "What are my AWS costs and optimization opportunities?",
                "metadata": {"source": "websocket_demo"},
                "timestamp": datetime.now().isoformat(),
                "message_id": "demo_msg_1"
            }
            
            print("📤 Sending message...")
            await websocket.send(json.dumps(test_message))
            
            # Listen for responses
            print("👂 Listening for responses...")
            async for message in websocket:
                data = json.loads(message)
                print(f"📨 Received: {data}")
                
                if data.get("event") == "execution_completed":
                    print("✅ Execution completed!")
                    break
                    
    except Exception as e:
        print(f"❌ WebSocket demo failed: {e}")
        print("💡 Make sure the server is running: python main.py")

if __name__ == "__main__":
    asyncio.run(websocket_demo())
    '''
    
    # Write temporary WebSocket client
    with open("temp_ws_client.py", "w") as f:
        f.write(websocket_client_code)
    
    try:
        subprocess.run(["python", "temp_ws_client.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ WebSocket client failed with exit code {e.returncode}")
    finally:
        # Clean up temp file
        Path("temp_ws_client.py").unlink(missing_ok=True)


def start_production_server():
    """Start the production server"""
    print("🏭 Starting Production Server")
    print("This will start the full system with all services")
    print()
    
    print("🔧 Checking docker-compose...")
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return
    
    try:
        # Start all services
        print("🚀 Starting all services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        print("✅ Services started!")
        print("🌐 API: http://localhost:8000")
        print("📊 Health: http://localhost:8000/health") 
        print("📖 Docs: http://localhost:8000/docs")
        print("🔌 WebSocket: ws://localhost:8000/api/v1/ws/chat")
        print("\n💡 Use 'docker-compose logs -f' to view logs")
        print("💡 Use 'docker-compose down' to stop services")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker and Docker Compose")


def show_system_info():
    """Show information about the system"""
    print("📋 FinOps Multi-Agent System Information")
    print("=" * 50)
    print()
    
    print("🏗️  System Architecture:")
    print("   • Phase 4 Multi-Agent Framework")
    print("   • Intelligent Agent Routing")
    print("   • Real-time WebSocket Communication") 
    print("   • Memory-Enhanced Processing")
    print("   • Cross-Domain Orchestration")
    print()
    
    print("🤖 Available Agents:")
    print("   • FinOps Agent: Financial modeling, cost analysis")
    print("   • GitHub Agent: Security analysis, vulnerability scanning")
    print("   • Document Agent: PDF processing, content extraction")
    print("   • Research Agent: Web research, fact verification")
    print("   • Deep Research Agent: Multi-agent coordination")
    print()
    
    print("🛠️  Technology Stack:")
    print("   • Backend: FastAPI + WebSocket")
    print("   • Agents: LangGraph + Custom Framework")
    print("   • Memory: Mem0 + Neo4j + Qdrant")
    print("   • Database: PostgreSQL + Redis")
    print("   • Frontend: HTML5 + WebSocket")
    print()
    
    print("📁 Key Files:")
    print("   • main.py - FastAPI application")
    print("   • demo.html - Web interface")
    print("   • demo_integrated.py - System demo")
    print("   • src/agents/ - Agent implementations")
    print("   • src/websocket/ - WebSocket handling")
    print("   • tests/ - Comprehensive test suite")


def main():
    parser = argparse.ArgumentParser(
        description="FinOps Multi-Agent System Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --integrated    # Run integrated system demo
  python run_demo.py --web          # Start web interface demo
  python run_demo.py --websocket    # Test WebSocket client
  python run_demo.py --production   # Start production server
  python run_demo.py --info         # Show system information
        """
    )
    
    parser.add_argument("--integrated", action="store_true", 
                       help="Run integrated system demo")
    parser.add_argument("--web", action="store_true",
                       help="Start web interface demo") 
    parser.add_argument("--websocket", action="store_true",
                       help="Run WebSocket client demo")
    parser.add_argument("--production", action="store_true",
                       help="Start production server with Docker")
    parser.add_argument("--info", action="store_true",
                       help="Show system information")
    
    args = parser.parse_args()
    
    if args.integrated:
        run_integrated_demo()
    elif args.web:
        run_web_demo()
    elif args.websocket:
        run_websocket_client()
    elif args.production:
        start_production_server()
    elif args.info:
        show_system_info()
    else:
        # Default: show options
        print("🎉 FinOps Multi-Agent System Demo Runner")
        print("=" * 50)
        print()
        print("Choose a demo option:")
        print("  🧪 --integrated    : Test real agent system (recommended)")
        print("  🌐 --web          : Web interface with server")
        print("  🔌 --websocket    : WebSocket client test")
        print("  🏭 --production   : Full Docker deployment")
        print("  📋 --info         : System information")
        print()
        print("Example: python run_demo.py --integrated")
        print("For help: python run_demo.py --help")


if __name__ == "__main__":
    main()