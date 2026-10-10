import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette import status

logger = logging.getLogger("dealersflow.security")

# Maximum permitted payload size: 25 Megabytes
MAX_PAYLOAD_BYTES = 25 * 1024 * 1024


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Defensive security middleware: Inspects Content-Length headers
    and immediately rejects uploads exceeding 25MB with HTTP 413.
    Prevents denial-of-service memory exhaustion attacks.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                length_bytes = int(content_length)
                if length_bytes > MAX_PAYLOAD_BYTES:
                    logger.warning(
                        f"[SECURITY ALERT] Blocked oversized upload: {length_bytes / (1024*1024):.1f} MB "
                        f"(Limit: {MAX_PAYLOAD_BYTES / (1024*1024)} MB)"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Payload too large. Maximum permitted file size is 25MB."
                        }
                    )
            except ValueError:
                pass

        return await call_next(request)