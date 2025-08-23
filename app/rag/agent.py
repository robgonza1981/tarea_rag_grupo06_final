# app/rag/agent.py
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, initialize_agent, Tool
from app.rag.retriever import QdrantRetriever
from app.settings import settings

def build_agent() -> AgentExecutor:
    retriever = QdrantRetriever(top_k=settings.TOP_K)

    def search_tool(q: str) -> str:
        hits = retriever.get(q)
        # devuelves texto plano concatenado para consumo del agente
        chunks: List[str] = []
        for h in hits:
            p = getattr(h, "payload", {}) or {}
            txt = (p.get("text") or "").strip()
            src = p.get("source", "desconocido")
            chunks.append(f"[{src}] {txt}")
        return "\n\n".join(chunks) or "Sin resultados relevantes."

    tools = [
        Tool(
            name="retriever_search",
            func=search_tool,
            description="Busca contexto en Qdrant y devuelve fragmentos relevantes.",
        )
    ]

    llm = ChatOpenAI(model=settings.GEN_MODEL, temperature=0)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False,
    )
    return agent
