# app/main.py
import json
import logging
import time
import uuid

from fastapi import FastAPI, Depends, Request

from app.api.routes_health import router as health_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_preview import router as preview_router
from app.api.routes_ready import router as ready_router
from app.core.auth import require_api_key

# ----------------------------
# Logging (structured JSON)
# ----------------------------
logger = logging.getLogger("borrower360")
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Borrower360 Underwriter Lab API")


# ----------------------------
# Middleware: Request ID + Logs
# ----------------------------
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    # If caller provides x-request-id, keep it; else generate one
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)

    # Add request id to every response
    response.headers["x-request-id"] = request_id

    # Structured log line (good for CloudWatch / ELK / Datadog)
    log_line = {
        "msg": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "client": request.client.host if request.client else None,
    }
    logger.info(json.dumps(log_line))

    return response


# ----------------------------
# Routers
# ----------------------------
# Usually health + readiness stay open
app.include_router(health_router)
app.include_router(ready_router)

# Protect data access endpoints with API key
app.include_router(datasets_router, dependencies=[Depends(require_api_key)])
app.include_router(preview_router, dependencies=[Depends(require_api_key)])


