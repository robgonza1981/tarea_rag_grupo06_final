# app/main.py
from app.rag.chains import build_chain
from app.services.rag_service import RAGService

if __name__ == "__main__":
    chain = build_chain()
    service = RAGService(chain)

    print("RAG CLI — escribe tu pregunta (sal con Ctrl+C)")
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        out = service.ask(q, debug=True)  # en CLI muestro debug para ver los hits
        print("\nRespuesta:\n", out["answer"])
        if out.get("hits"):
            print("\n[DEBUG] Hits:")
            for h in out["hits"]:
                print(f"  - {h.get('source')} | score={h.get('score'):.3f}")
