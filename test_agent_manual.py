
import asyncio
import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

from app.services.agent import chat_service

async def test_agent():
    print("--- Testing Agent ---")
    
    # Test 1: Chitchat
    print("\nQuery: Hello!")
    try:
        response = await chat_service("Hello! Who are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: GitHub Intent (Simulated query)
    print("\nHow much project GitHub do you have?")
    try:
        response = await chat_service("How much project GitHub do you have?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: CV Intent (Simulated query)
    print("\nWhat are your skills?")
    try:
        response = await chat_service("What are your skills?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())
