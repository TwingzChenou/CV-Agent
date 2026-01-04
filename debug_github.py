import os
import asyncio
from llama_index.readers.github import GithubClient
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

async def main():
    if token:
        client = GithubClient(github_token=token)
        print("Endpoints:", client._endpoints.keys())
        try:
            # Let's try getBranch since we know the repo
            print("Testing getBranch...")
            commit = await client.get_branch("TwingzChenou", "CV-Agent", "main")
            print("Branch info found, commit SHA:", commit)
            print("GitHub Connection Valid!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
