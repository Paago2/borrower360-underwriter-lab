# app/core/auth.py
from __future__ import annotations

import os
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader


# This defines the OpenAPI security scheme => Swagger gets an "Authorize" button.
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_scheme)) -> str:
    """
    Enterprise-style API key check.
    - Client must send: X-API-Key: <value>
    - Expected value comes from env var API_KEY
    """
    expected = os.environ.get("API_KEY")
    if not expected:
        # Force explicit config so don't "accidentally" run unsecured.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not set on the server. Export API_KEY before starting uvicorn.",
        )

    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key (use X-API-Key header).",
        )

    return api_key
