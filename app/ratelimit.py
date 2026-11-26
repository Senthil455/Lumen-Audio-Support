"""Per-caller rate limiting.

Keyed by caller, not by endpoint, and the key is derived from the forwarded IP
when the app sits behind a proxy - otherwise every request looks like it comes
from the load balancer and one noisy client locks everyone out.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.logging_setup import caller_id


def rate_limit_key(request: Request) -> str:
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return caller_id(request) or get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key, headers_enabled=True)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    detail = getattr(exc, "detail", "rate limit exceeded")
    headers = {"Retry-After": "60"}
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Slow down and try again shortly.",
            "limit": str(detail),
        },
        headers=headers,
    )
