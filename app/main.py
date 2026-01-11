# app/main.py
from fastapi import FastAPI, Depends

from app.api.routes_health import router as health_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_preview import router as preview_router
from app.core.auth import require_api_key

app = FastAPI(title="Borrower360 Underwriter Lab API")

# Usually health stays open
app.include_router(health_router)

# Protect data access endpoints with API key
app.include_router(datasets_router, dependencies=[Depends(require_api_key)])
app.include_router(preview_router, dependencies=[Depends(require_api_key)])


