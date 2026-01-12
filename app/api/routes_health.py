from fastapi import APIRouter
from app.core.paths import project_root

root = project_root()



router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

