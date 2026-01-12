from fastapi import APIRouter

router = APIRouter(tags=["default"])

@router.get("/ready")
def ready():
    # Later we’ll add checks: config loaded, registry ok, sanctions index ok
    return {"status": "ready"}
