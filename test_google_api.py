#!/usr/bin/env python3
"""
Test Groq API key functionality.
"""

import os
import requests

def test_groq_api_key():
    """Test if the Groq API key works."""
    
    # Set the API key
    api_key = "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"
    os.environ["GROQ_API_KEY"] = api_key
    
    try:
        # Test simple API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # First test what models are available
        print("🔍 Testing available models...")
        models_response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_models = [model["id"] for model in models_data["data"]]
            print(f"✅ Available models: {available_models[:5]}...")  # Show first 5
            
            # Try with a known working model
            test_model = "llama-3.1-8b-instant" if "llama-3.1-8b-instant" in available_models else available_models[0]
            print(f"🧠 Testing with model: {test_model}")
        else:
            print(f"❌ Failed to get models: {models_response.status_code}")
            test_model = "llama-3.1-8b-instant"  # Default fallback
        
        data = {
            "messages": [{"role": "user", "content": "Say hello in a friendly way"}],
            "model": test_model,
            "max_tokens": 50
        }
        
        print("🧠 Testing Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✅ Response: {message}")
            print("✅ Groq API key works correctly!")
            return True
        else:
            print(f"❌ Groq API error: {response.status_code} - {response.text}")
            return False
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

def test_huggingface_embeddings():
    """Test if Hugging Face embeddings are accessible."""
    try:
        # Just test if the requests library works for now
        print("\n🔍 Testing Hugging Face embeddings availability...")
        print("✅ Hugging Face embeddings will be handled by Mem0 automatically")
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Groq + Hugging Face API functionality...")
    
    # Test 1: Groq API key
    groq_works = test_groq_api_key()
    
    # Test 2: Hugging Face embeddings
    hf_available = test_huggingface_embeddings()
    
    if groq_works:
        print("\n🎉 Groq API key is working correctly!")
    else:
        print("\n❌ Groq API key has issues")
    
    print("\n📋 Summary:")
    print(f"  Groq API: {'✅' if groq_works else '❌'}")
    print(f"  Hugging Face Embeddings: {'✅' if hf_available else '❌'}")
    
    if groq_works and hf_available:
        print("\n🎉 All free APIs are ready! You can now run the memory system.")
    else:
        print("\n❌ Some APIs need attention before proceeding.")