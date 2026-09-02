"""Command-line entry point for the minimal LangGraph medical-imaging agent."""

import logging
from pathlib import Path

from app.agents.graph import MedicalImagingAgent
from app.rag.siliconflow import SiliconFlowClient
from app.rag.workflow import MedicalRAGWorkflow


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge"
logger = logging.getLogger(__name__)


def main() -> None:
    client = SiliconFlowClient(str(PROJECT_DIR))
    rag_workflow = MedicalRAGWorkflow.from_knowledge_dir(client, KNOWLEDGE_DIR)
    agent = MedicalImagingAgent(rag_workflow=rag_workflow, llm_client=client)
    print("医学影像 Agent 已就绪。输入 exit 或 quit 可结束。")

    while True:
        question = input("\n医学影像问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        state = agent.invoke(question)
        print(f"\n意图：{state['intent']}")
        if state["retrieved_context"]:
            print("\n送入 LLM 的检索上下文：")
            print(state["retrieved_context"])
        print("\n回答：")
        print(state["final_answer"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
