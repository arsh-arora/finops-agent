"""
Simple test application with intentional vulnerabilities for security analysis testing.
"""

import os
import sqlite3
import random

# Intentional vulnerability: hardcoded credentials
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef"

def get_user_data(user_id):
    """Intentional SQL injection vulnerability for testing."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Vulnerable SQL query
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    
    return cursor.fetchall()

def generate_session_token():
    """Intentional weak random generation for testing."""
    # Using predictable random for session tokens
    random.seed(12345)
    return random.randint(1000, 9999)

def process_user_input(data):
    """Unvalidated input processing for testing."""
    # Direct execution without validation
    return eval(data)

if __name__ == "__main__":
    print("Test application starting...")
    token = generate_session_token()
    print(f"Session token: {token}")