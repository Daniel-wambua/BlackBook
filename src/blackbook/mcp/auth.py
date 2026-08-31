"""HTTP transport authentication for BlackBook.

The stdio transport (the default, used by Claude Code / Cursor / VS Code)
needs none of this: the client owns the process. But ``blackbook serve --http``
exposes the knowledge base — and the ``knowledge_context`` write path — to
whatever can reach the bound address. When a bearer token is configured, every
request to the HTTP server must present it; when it is not, binding a
non-loopback address is refused unless the operator explicitly opts out.

This is deliberately a simple shared-secret check, not the full MCP OAuth
resource-server machinery: BlackBook is a personal/local tool, and a static
token is the right size for that threat model.
"""

from __future__ import annotations

import secrets
from typing import Any

from blackbook.config import Settings, is_loopback_host


class RequireBearer:
    """Pure-ASGI middleware that enforces ``Authorization: Bearer <token>``.

    Mounted around the whole HTTP app (MCP endpoint, /health, and the landing
    page alike) so an unauthenticated party learns nothing but "401". Works
    with any Starlette/uvicorn app — no FastMCP internals touched.
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        auth = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                auth = value
                break
        expected = f"Bearer {self.token}".encode()
        if not secrets.compare_digest(auth, expected):
            await _send_unauthorized(send)
            return

        await self.app(scope, receive, send)


async def _send_unauthorized(send) -> None:
    """Emit a bare 401 with no detail that would leak why it was refused."""
    body = b'{"error":"unauthorized"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="blackbook"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def enforce_bind_safety(settings: Settings) -> str | None:
    """Refuse unsafe HTTP binds before the socket is ever opened.

    Returns a warning string when the bind is unusual but allowed, or raises
    ``RuntimeError`` when a non-loopback bind has no auth token configured
    (unless ``server.require_auth_off_loopback`` was explicitly disabled).

    Called by both entrypoints (``python -m blackbook.server`` and
    ``blackbook serve --http``) before uvicorn starts.
    """
    host = settings.server.host
    token = (settings.server.auth_token or "").strip()
    loopback = is_loopback_host(host)
    if not loopback:
        if not token and settings.server.require_auth_off_loopback:
            raise RuntimeError(
                f"Refusing to bind {host!r} without authentication: the HTTP "
                "transport exposes the whole knowledge base and the "
                "knowledge_context write path to every host that can reach "
                "this address. Set server.auth_token in your config (and "
                "connect clients with the same bearer token), or bind "
                "127.0.0.1. To accept the exposure deliberately, set "
                "server.require_auth_off_loopback: false."
            )
        if not token:
            return (
                f"WARNING: binding {host!r} without an auth token — the "
                "knowledge base is readable (and knowledge_context writable) "
                "by any host that can reach this address."
            )
    return None


def wrap_http_app(app, settings: Settings):
    """Wrap a Starlette app with bearer auth when a token is configured.

    Returns the app unchanged when no token is set (stdio and loopback-only
    setups pay nothing).
    """
    token = (settings.server.auth_token or "").strip()
    if not token:
        return app
    return RequireBearer(app, token)
