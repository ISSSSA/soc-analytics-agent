from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class BearerAuth:
    """Bearer-token verifier; auth disabled when api_key is None (dev mode)."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        if api_key is None:
            logger.warning(
                "INFERENCE_API_KEY is not set; authentication is DISABLED. "
                "Do not deploy this way in production.",
            )

    async def __call__(self, request: Request) -> None:
        if self._api_key is None:
            return

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header (expected: 'Bearer <token>')",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = header[7:].strip()
        if not secrets.compare_digest(token, self._api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
