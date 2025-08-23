# ---------- Variables ----------
PY=python
DOCS_DIR=./data/docs
COLL_RECURSIVE=docs_recursive
COLL_SEMANTIC=docs_semantic
CHUNK_SIZE?=1000
CHUNK_OVERLAP?=200
SEM_THRESHOLD_TYPE?=percentile  # percentile | standard_deviation
SEM_THRESHOLD_AMOUNT?=95        # más bajo => más cortes
PORT?=8000

# ---------- Ayuda ----------
.PHONY: help
help:
	@echo "Targets disponibles:"
	@echo "  install            - Instala dependencias"
	@echo "  run                - Levanta API FastAPI (/ask, /ask_debug, /qdrant/stats)"
	@echo "  run-langserve      - Levanta LangServe (/rag, /rag_debug y playgrounds)"
	@echo "  index-recursive    - Indexa usando RecursiveCharacterTextSplitter"
	@echo "  index-semantic     - Indexa usando SemanticChunker (umbral configurable)"
	@echo "  stats              - Muestra stats de Qdrant desde la API local"
	@echo "  eval-recursive     - Evalúa colección recursive y genera CSV"
	@echo "  eval-semantic      - Evalúa colección semantic y genera CSV"
	@echo "  fly-secrets        - (Guía) Setea secrets en Fly.io"
	@echo "  fly-deploy         - Despliega a Fly.io"

# ---------- Setup ----------
.PHONY: install
install:
	pip install -r requirements.txt

# ---------- Run local ----------
.PHONY: run
run:
	uvicorn app.server:app --reload --host 0.0.0.0 --port $(PORT)

.PHONY: run-langserve
run-langserve:
	uvicorn app.server_langserve:app --reload --host 0.0.0.0 --port $(PORT)

# ---------- Indexación ----------
.PHONY: index-recursive
index-recursive:
	$(PY) -m app.ingest.index_qdrant $(DOCS_DIR) \
		--strategy recursive \
		--chunk-size $(CHUNK_SIZE) --chunk-overlap $(CHUNK_OVERLAP) \
		--collection $(COLL_RECURSIVE) --recreate

.PHONY: index-semantic
index-semantic:
	$(PY) -m app.ingest.index_qdrant $(DOCS_DIR) \
		--strategy semantic \
		--sem-threshold-type $(SEM_THRESHOLD_TYPE) \
		--sem-threshold-amount $(SEM_THRESHOLD_AMOUNT) \
		--collection $(COLL_SEMANTIC) --recreate

# ---------- Qdrant stats ----------
.PHONY: stats
stats:
	curl -s http://localhost:$(PORT)/qdrant/stats | jq .

# ---------- Evaluación ----------
.PHONY: eval-recursive
eval-recursive:
	$(PY) -m app.eval.evaluator $(COLL_RECURSIVE) app/eval/questions_answerable.json app/eval/questions_unanswerable.json

.PHONY: eval-semantic
eval-semantic:
	$(PY) -m app.eval.evaluator $(COLL_SEMANTIC) app/eval/questions_answerable.json app/eval/questions_unanswerable.json

# ---------- Fly.io ----------
.PHONY: fly-secrets
fly-secrets:
	@echo "Recuerda: .env NO se sube. Usa secrets en Fly.io:"
	@echo "  fly secrets set OPENAI_API_KEY=sk-... QDRANT_URL=https://... QDRANT_API_KEY=..."
	@echo "  fly secrets set QDRANT_COLLECTION=$(COLL_RECURSIVE) EMBED_MODEL=text-embedding-3-small EMBED_DIM=1536"
	@echo "  fly secrets set GEN_MODEL=gpt-4.1-mini CHUNK_STRATEGY=recursive CHUNK_SIZE=$(CHUNK_SIZE) CHUNK_OVERLAP=$(CHUNK_OVERLAP)"

.PHONY: fly-deploy
fly-deploy:
	fly deploy