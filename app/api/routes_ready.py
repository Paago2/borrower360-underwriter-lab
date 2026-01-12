from fastapi import APIRouter, Response, status
from app.services.readiness import check_readiness

from app.core.paths import project_root

root = project_root()


router = APIRouter(tags=["default"])

@router.get("/ready")
def ready(response: Response):
    result = check_readiness()
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "checks": result.checks}
    return {"status": "ready", "checks": result.checks}
