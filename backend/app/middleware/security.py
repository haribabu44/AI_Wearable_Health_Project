"""
middleware/security.py
----------------------
Two responsibilities:
  1. API key authentication  — checked on every /api/* request
  2. In-process rate limiter — sliding window per client IP
     (For production, replace with Redis-backed slowapi or nginx rate limiting)
"""

import time
import logging
from collections import defaultdict, deque
from typing import Deque

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combines API key check + sliding-window rate limiter."""

    def __init__(self, app, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self.exempt_paths: list[str] = exempt_paths or ["/", "/health", "/docs",
                                                         "/openapi.json", "/redoc"]
        # {ip: deque of request timestamps}
        self._windows: defaultdict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Skip security checks for non-API routes
        if any(path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        # ── API key check ────────────────────────────────────────────────
        api_key = request.headers.get(settings.api_key_header)
        if api_key != settings.api_key:
            logger.warning("Rejected request — invalid API key (path=%s)", path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key. "
                                   f"Set the '{settings.api_key_header}' header."},
            )

        # ── Rate limiting ────────────────────────────────────────────────
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._windows[client_ip]

        # Evict timestamps outside the rolling window
        while window and window[0] < now - settings.rate_limit_window:
            window.popleft()

        if len(window) >= settings.rate_limit_requests:
            retry_after = int(settings.rate_limit_window - (now - window[0])) + 1
            logger.warning(
                "Rate limit hit for %s (%d req/%ds)",
                client_ip, settings.rate_limit_requests, settings.rate_limit_window,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after)},
                content={
                    "detail": f"Rate limit exceeded. "
                              f"Max {settings.rate_limit_requests} requests "
                              f"per {settings.rate_limit_window}s.",
                    "retry_after_seconds": retry_after,
                },
            )

        window.append(now)
        return await call_next(request)
