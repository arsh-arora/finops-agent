#!/usr/bin/env python3
"""
Simple Demo - Just run the actual production system
"""

import asyncio
import json
import websockets
from datetime import datetime


async def test_websocket_connection():
    """Test WebSocket connection to the running server"""
    
    print("🔌 Testing WebSocket connection to production system")
    print("=" * 60)
    
    # First, get a JWT token
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            # Get authentication token
            print("🔑 Getting authentication token...")
            response = await client.post(
                "http://localhost:8000/auth/token",
                params={"user_id": "demo_user"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data["access_token"]
                print(f"✅ Got token: {token[:50]}...")
            else:
                print(f"❌ Failed to get token: {response.text}")
                return
    
    except Exception as e:
        print(f"❌ Failed to get token: {e}")
        print("💡 Make sure the server is running: python main.py")
        return
    
    # Connect to WebSocket
    try:
        uri = f"ws://localhost:8000/api/v1/ws/chat?token={token}"
        print(f"🚀 Connecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Test messages for different agents
            test_messages = [
                "What are my AWS costs and how can I optimize them?",
                "Analyze security vulnerabilities in my GitHub repository", 
                "Extract data from this financial PDF document",
                "Research latest cloud cost optimization trends",
                "Coordinate analysis across financial, security, and research domains"
            ]
            
            for i, message_text in enumerate(test_messages, 1):
                print(f"\n📤 Test {i}: {message_text}")
                
                # Send message
                message = {
                    "type": "user_message",
                    "text": message_text,
                    "metadata": {"source": "demo"},
                    "timestamp": datetime.now().isoformat(),
                    "message_id": f"demo_msg_{i}"
                }
                
                await websocket.send(json.dumps(message))
                print("   📨 Message sent")
                
                # Listen for responses
                print("   👂 Listening for responses...")
                response_count = 0
                
                while response_count < 1000:  # Limit responses per message
                    try:
                        print(f"   🕐 Waiting for response (attempt {response_count + 1})...")
                        response = await asyncio.wait_for(websocket.recv(), timeout=3000.0)  # Increased timeout
                        data = json.loads(response)
                        
                        event_type = data.get("event", "message")
                        
                        # Show full response for agent responses
                        if event_type == "agent_response":
                            agent_message = data.get("message", "")
                            print(f"   🤖 AGENT RESPONSE:")
                            print(f"      {agent_message}")
                        else:
                            print(f"   📡 {event_type}: {str(data)}...")
                        
                        response_count += 1
                        
                        # Stop listening after completion
                        if event_type in ["execution_completed", "execution_failed"]:
                            break
                            
                    # except asyncio.TimeoutError:
                    #     print("   ⏱️  Response timeout - moving to next message")
                    #     break
                    except Exception as e:
                        print(f"   ❌ Error receiving: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                
                # Small pause between messages
                await asyncio.sleep(2)
                
    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket connection closed")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


async def main():
    """Main demo function"""
    print("🎉 FinOps Multi-Agent System - Simple Production Demo")
    print("=" * 70)
    print()
    print("This demo connects to the running production system and tests all agents.")
    print("Make sure the server is running first: python main.py")
    print()
    
    await test_websocket_connection()
    
    print("\n🎉 Demo completed!")
    print("\n💡 To see the web interface:")
    print("   1. Make sure server is running: python main.py")
    print("   2. Open demo.html in your browser")
    print("   3. Test the interactive chat interface")


if __name__ == "__main__":
    asyncio.run(main())