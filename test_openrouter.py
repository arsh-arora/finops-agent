#!/usr/bin/env python3
"""
Quick test of OpenRouter integration
"""

import asyncio
from src.llm.openrouter_client import OpenRouterClient


async def test_openrouter():
    """Test OpenRouter client"""
    print("🧪 Testing OpenRouter Client")
    print("=" * 40)
    
    client = OpenRouterClient()
    
    test_messages = [
        "What are my AWS costs and how can I optimize them?",
        "Analyze security vulnerabilities in my GitHub repository",
        "Extract data from this PDF document",
        "Research cloud cost optimization trends",
        "Coordinate analysis across financial, security, and research domains"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🔍 Test {i}: {message}")
        
        try:
            response = await client.complete([
                {"role": "user", "content": f"You are an intelligent agent router. Analyze this request and determine which agent domain it belongs to: {message}"}
            ])
            
            print(f"   ✅ Response: {response}")
            
        except Exception as e:
            print(f"   ⚠️  Fallback used: {e}")
            # Test fallback
            response = await client._fallback_routing(message)
            print(f"   🔄 Fallback response: {response}")
    
    print(f"\n📊 Total API calls made: {client.call_count}")


if __name__ == "__main__":
    asyncio.run(test_openrouter())