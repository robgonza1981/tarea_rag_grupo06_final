# app/api/routes/__init__.py
from fastapi import APIRouter
from .ask import router as ask_router
from .qdrant import router as qdrant_router  

router = APIRouter()
router.include_router(ask_router)
router.include_router(qdrant_router) 