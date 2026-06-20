import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

backend_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=backend_root / ".env")
sys.path.insert(0, str(backend_root))

from app.engine.generate import agent

async def main():
    query = "Quelles sont les disponibilités de Quentin ?"
    agent_input = (
        f"{query}\n"
        f"### DIRECTIVE DE CONTRÔLE ###\n"
        f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
        f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
        f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
    )
    
    response = await agent.run(agent_input)
    print("RESPONSE TEXT:", response.response.content if hasattr(response.response, 'content') else response.response)
    print("TOOL CALLS TYPE:", type(response.tool_calls))
    print("TOOL CALLS:")
    for i, tc in enumerate(response.tool_calls):
        print(f"Tool Call {i}:")
        print("  Type:", type(tc))
        print("  Directory:", dir(tc))
        print("  String repr:", str(tc))
        if hasattr(tc, 'tool_name'):
            print("  tool_name:", tc.tool_name)
        if hasattr(tc, 'tool_kwargs'):
            print("  tool_kwargs:", tc.tool_kwargs)

if __name__ == "__main__":
    asyncio.run(main())
