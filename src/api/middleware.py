"""FastAPI middleware: per-request logging and global exception handler."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_request_log = logging.getLogger("api.requests")
_error_log = logging.getLogger("api.errors")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and duration for every request."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Process the request, attach X-Process-Time header, and emit a log line."""
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"

        msg = f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)"
        if response.status_code < 400:
            _request_log.info(msg)
        elif response.status_code < 500:
            _request_log.warning(msg)
        else:
            _request_log.error(msg)

        return response


def register_exception_handler(app: FastAPI) -> None:
    """Register a catch-all 500 handler that logs the full traceback."""

    @app.exception_handler(Exception)
    async def _handle(request: Request, exc: Exception) -> JSONResponse:
        _error_log.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "error_type": type(exc).__name__,
                "path": str(request.url.path),
            },
        )
