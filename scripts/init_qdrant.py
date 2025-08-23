import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Cargar variables del .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Conexión a Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Crear o recrear la colección
client.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=rest.VectorParams(
        size=1536,  # tamaño del embedding que uses
        distance=rest.Distance.COSINE
    )
)

print(f"✅ Colección '{QDRANT_COLLECTION}' creada o actualizada correctamente.")
