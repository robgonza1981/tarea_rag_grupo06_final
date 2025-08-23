# app/server.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.api.routes import router as api_router

app = FastAPI(title="FEN RAG API", version="0.1.0")

# Redirección para que "Swagger solito" cargue al abrir /
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Infra"])
def health():
    return {"ok": True}

# Monta las rutas de la API: /ask, /ask_debug, /qdrant/stats
app.include_router(api_router)
