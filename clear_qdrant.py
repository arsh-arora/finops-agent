#!/usr/bin/env python3
"""
Clear Qdrant collection to fix dimension mismatch.
"""

import requests

def clear_qdrant_collection():
    """Delete the existing collection so it can be recreated with correct dimensions."""
    
    try:
        print("🧹 Clearing Qdrant collection 'finops_memories'...")
        
        # Delete the collection
        response = requests.delete(
            "http://localhost:6333/collections/finops_memories",
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Successfully deleted collection 'finops_memories'")
            print("✅ Next run will create collection with correct 384 dimensions")
            return True
        elif response.status_code == 404:
            print("✅ Collection 'finops_memories' doesn't exist - that's fine")
            return True
        else:
            print(f"❌ Failed to delete collection: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Clearing Qdrant collection to fix dimension mismatch...")
    success = clear_qdrant_collection()
    
    if success:
        print("\n🎉 Ready to test memory system again!")
        print("Run: python scripts/smoke_test_memory.py")
    else:
        print("\n❌ Manual cleanup may be needed")
        print("You can also restart Docker containers to reset Qdrant")