FEN · RAG (Ventas / Compras / RR.HH.)

Asistente empresarial con RAG (Retrieval Augmented Generation) sobre procedimientos de Ventas/Compras/RR.HH., construido con FastAPI + LangServe, Qdrant (vector DB) y OpenAI para embeddings y generación.
Incluye chunking jerárquico, normalización de siglas (OV, NC, OC, DTE), smalltalk instantáneo y sugerencias validadas según la base.

Demo local

UI: http://localhost:8000/ui

Playgrounds LangServe:

http://localhost:8000/rag/playground

http://localhost:8000/chat/playground

Swagger: http://localhost:8000/docs

Health: http://localhost:8000/health

Tabla de contenidos

Arquitectura

Estructura del proyecto

Requisitos

Configuración (local, Python 3.12)

Cargar/actualizar Qdrant

Ejecutar el servicio

Evaluación (datasets + 2 evaluadores)

Cómo probar (ejemplos)

Decisiones de diseño

Despliegue (Flyio)

Checklist de entrega

Troubleshooting

Arquitectura
┌────────┐   pregunta             ┌─────────────┐  embeddings  ┌─────────┐
│  UI/   │ ─────────────────────► │  LangServe  │ ───────────► │ Qdrant  │
│ FastAPI│                        │ (Runnables) │              └─────────┘
└────────┘   respuesta            └─────┬───────┘     ▲
                                         │            │
                                         ▼            │
                                    OpenAI Chat ◄─────┘


RAG: recupera top_k chunks desde Qdrant (COSINE), arma un contexto y el LLM responde solo si hay soporte documental (si no, ABSTAIN).

Smalltalk: saludos/despedidas/agradecimientos sin ir a Qdrant (respuesta inmediata y cálida).

Sugerencias: catálogo reducido, validado al iniciar con la base (cache) para no sugerir cosas inexistentes.

Chunking: secciones → micro-chunks (~900 chars, overlap 150) + metadata jerárquica.

Estructura del proyecto
app/
  server_langserve.py        # FastAPI + LangServe + UI + rutas
  rag/
    chains.py                # RAG runnable, normalización OV/NC/OC/DTE, ABSTAIN
    tools.py                 # (opcional) agente
  ingest/
    chunking.py              # chunking jerárquico
    index_qdrant.py          # ingesta a Qdrant (create/recreate + upsert)
    embedders.py             # OpenAIEmbedder
  ui/
    index.html               # UI ligera de chat
data/
  docs/                      # .txt/.md con la base de conocimiento
  eval/
    respondibles.jsonl       # preguntas respondibles (dataset 1)
    no_respondibles.jsonl    # preguntas NO respondibles (dataset 2)
.env                         # credenciales locales (no subir)
requirements.txt             # dependencias

Requisitos

Python 3.12

Cuenta OpenAI (API key)

Qdrant Cloud o Qdrant self-hosted

Configuración (local, Python 3.12)
# 0) (opcional) borrar venv anterior
# Windows PowerShell:
Remove-Item -Recurse -Force .venv 2>$null
# macOS/Linux:
rm -rf .venv

# 1) crear venv en Python 3.12
py -3.12 -m venv .venv         # Windows
# o: python3.12 -m venv .venv   # macOS/Linux

# 2) activar
.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate    # macOS/Linux

# 3) instalar deps
pip install -U pip
pip install -r requirements.txt


Variables de entorno (archivo .env para local; en producción usar secrets):

OPENAI_API_KEY=sk-...
QDRANT_URL=https://XXXXXXXXXXX.qdrant.cloud
QDRANT_API_KEY=...
QDRANT_COLLECTION=document_embeddings

GEN_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536

TOP_K=5
MIN_SCORE=0.0
MAX_CONTEXT_CHARS=4000

Cargar/actualizar Qdrant

Coloca tus .txt/.md en data/docs/.

Ejecuta la ingesta (recrea la colección y sube embeddings + metadata):

python -m app.ingest.index_qdrant --path .\data\docs\ --collection document_embeddings --recreate 1


Se crean índices HNSW y de payload; cada chunk lleva metadata:
source, doc_id, doc_title, section_title, section_path, position, n_chars, ts_ingested, text.

Ejecutar el servicio
uvicorn app.server_langserve:app --reload --port 8000 --env-file .env


UI: http://localhost:8000/ui

Playgrounds: http://localhost:8000/rag/playground · http://localhost:8000/chat/playground

Swagger: http://localhost:8000/docs

Health: http://localhost:8000/health

Evaluación (datasets + 2 evaluadores)

En data/eval/ están tus respondibles y no respondibles. Tienes además dos evaluadores (ej.: uno “contains/heurístico” para respondibles y otro que verifica ABSTAIN exacto para no respondibles).

Si tus scripts tienen otro nombre, ajusta el módulo/ruta en los comandos.

# 1) Eval respondibles: se espera que la respuesta contenga ciertos tokens
python -m app.eval.eval_contains --file data/eval/respondibles.jsonl --endpoint http://localhost:8000/rag/invoke

# 2) Eval no respondibles: se espera EXACTAMENTE el string ABSTAIN
python -m app.eval.eval_abstain --file data/eval/no_respondibles.jsonl --endpoint http://localhost:8000/rag/invoke


ABSTAIN canónico:

No tengo información suficiente en los documentos para responder con certeza.

Cómo probar (ejemplos)

Smalltalk (instantáneo): “hola”, “hola cómo estás?”, “gracias”, “chao”.

Consultas típicas (según tu base):

“¿Dónde se carga el archivo con toda la documentación del pedido?”

“¿En qué módulo se imprime la factura interna (IVN)?”

“¿Cómo generar una OV negativa y la NC tras rechazo sin devolución?”

No respondibles (debe ABSTAIN): “¿Cómo configuro SAP?”, “¿Política de vacaciones 2022?”…

Si una pregunta no sale y sabes que está en los docs:

sube TOP_K (p. ej. 5),

revisa si el chunk quedó partido,

amplía normalizaciones (e.g., “duplicar OV”, “copia de OV”, etc.).

Decisiones de diseño
Chunking jerárquico

Detecta secciones (títulos en mayúsculas/numéricas/markdown).

Divide en micro-chunks (~900 chars, overlap 150).

Metadata rica por chunk: jerarquía section_path, doc_title, source, etc.
Facilita mejor recuperación y futuros filtros/reranking.

Recuperación + generación

Qdrant (COSINE) con TOP_K/MIN_SCORE configurables.

Normalización de consulta (regex con límites de palabra) para siglas/typos:
OV→“orden de venta”, NC→“nota de crédito”, OC, DTE, “nota de credi*”.

Prompt “seguro”: si no hay soporte, responde ABSTAIN exacto.

Smalltalk + sugerencias

Smalltalk no toca Qdrant → latencia mínima.

Sugerencias validadas: se cachean al arrancar para no sugerir imposibles y se muestran 3 (rotadas) al abstener.

Despliegue (Fly.io)
# 1) lanzar app (si no existe)
# flyctl launch --no-deploy

# 2) secrets
flyctl secrets set \
  OPENAI_API_KEY=sk-... \
  QDRANT_URL=... \
  QDRANT_API_KEY=... \
  QDRANT_COLLECTION=document_embeddings \
  GEN_MODEL=gpt-4o-mini \
  EMBED_MODEL=text-embedding-3-small \
  EMBED_DIM=1536 \
  TOP_K=5 \
  MIN_SCORE=0.0 \
  MAX_CONTEXT_CHARS=4000

# 3) build & deploy
flyctl deploy

#### RESULTADOS DE EVALUACIONES 
 RESULTADOS DE LA PRUEBA - Estrategia Chunk Semántico
Puntuación General: 9.29/10
Precisión: 9.43/10
Relevancia: 9.80/10
Completitud: 8.47/10
Conciencia de Contexto: 9.47/10

RESULTADOS DE LA PRUEBA - Estrategia Chunk Recursivo Jerárquico
Puntuación General: 9.39/10
Precisión: 9.50/10
Relevancia: 9.73/10
Completitud: 8.80/10
Conciencia de Contexto: 9.53/10

### CONCLUSIONES ELECCION ESTRATEGIA DE CHUNKING
  Se escogió la estrategia de  chunking jerárquico con metadata porque los  documentos no son “párrafos sueltos”, son procedimientos de ERP (QAD) con títulos, rutas de menú (7.12.10, 3.6.30.31), pasos y secciones. Eso hace que el chunking jerárquico gane por: 
  
  1.- Recuperación más precisa por tema/sección
  
    Cada chunk “hereda” doc_id, doc_title, section_title, section_path y position. Eso permite que el buscador y el reranker favorezcan la sección correcta (p. ej. “Liberación de OV”) en vez de fragmentos mezclados de otros temas.

  2.- Recomposición coherente del contexto 
    
    En chains._format_context() agrupamos por (doc_id, section_title) y ordenamos por position. Así el LLM ve un bloque continuo de pasos/códigos dentro de la misma sección (menú → validaciones → salida), en vez de saltos de documento, lo que reduce abstenciones y alucines.
  
  3.- Boosts de dominio más efectivos
    
    Las  consultas traen señales como “7.12.10”, “stock rápido”, “aprobación comercial”. Con metadata jerárquica el reranker sumó puntos cuando el chunk viene de una sección “correcta” y cuando el strategy=="hier". Eso es lo que usamos en _heuristic_rerank().

  4.- Códigos de menú preservados
    Los códigos (7.10.20.36, 3.6.30.31, etc.) suelen aparecer en encabezados o en líneas cercanas. Si chunkeamos  de manera plana, se separan del texto que los explica. Con jerarquía y position permanecen juntos → mejores respuestas “con código literal”.
  
  5.- Menos ruido en top-K
 
    El flat/semántico mete trozos similares pero de otras áreas (RR.HH., Compras) y el LLM se confunde. El jerárquico, al respetar secciones y orden, reduce la mezcla temática y mejora el answer_ok.

  
  6.-Sugerencias más fiables
    
    Tu quick_can_answer(strict=True) pide que el top-1 venga de strategy=="hier" y con señales fuertes (códigos/overlap). Esa condición evita sugerir cosas que luego el RAG no puede sostener.
  
  7.- Costo  y latencia
 
   Al agrupar y ordenar, se puede pasar menos texto redundante al LLM (contexto más “denso”), bajando tokens sin perder continuidad.

 
 En resumen: para este  caso, la metadata jerárquica permite recuperar la sección exacta, rearmar los pasos en orden, reforzar con códigos de menú y producir respuestas verificables y consistentes. Por eso subió el answer_ok en el evaluador y por eso resiste mejor las variaciones de la pregunta.