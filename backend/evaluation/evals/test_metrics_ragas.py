import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

backend_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=backend_root / ".env")
sys.path.insert(0, str(backend_root))

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ToolCallAccuracy, AgentGoalAccuracy
from ragas.messages import HumanMessage, AIMessage, ToolCall

async def main():
    print("🤖 Initializing Async Ollama client and Ragas LLM wrapper...")
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ragas_judge_llm = llm_factory("mistral", client=client)
    
    metric_tool = ToolCallAccuracy(strict_order=True)
    metric_goal = AgentGoalAccuracy(llm=ragas_judge_llm)
    
    # Sample 1: Perfect match
    print("\n--- Running Sample 1 (Perfect Match) ---")
    messages = [
        HumanMessage(content="Quelles sont les disponibilités de Quentin ?"),
        AIMessage(
            content="Quentin est disponible immédiatement.",
            tool_calls=[
                ToolCall(name="cv_query_engine", args={"input": "Quelles sont les disponibilités de Quentin ?"})
            ]
        )
    ]
    ref_tool_calls = [
        ToolCall(name="cv_query_engine", args={"input": "Quelles sont les disponibilités de Quentin ?"})
    ]
    ref_answer = "Quentin est disponible immédiatement."
    
    score_tool = await metric_tool.ascore(user_input=messages, reference_tool_calls=ref_tool_calls)
    print("Tool Call Accuracy Score:", score_tool.value)
    
    score_goal = await metric_goal.ascore(user_input=messages, reference=ref_answer)
    print("Agent Goal Accuracy Score:", score_goal.value)

if __name__ == "__main__":
    asyncio.run(main())
