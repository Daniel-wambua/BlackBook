"""Tests for HTTP-transport authentication and bind safety."""

from __future__ import annotations

import asyncio

import pytest

from blackbook.config import ServerConfig, Settings, is_loopback_host
from blackbook.mcp.auth import (
    RequireBearer,
    enforce_bind_safety,
    wrap_http_app,
)


def _settings(host: str, token: str = "", strict: bool = True) -> Settings:
    s = Settings()
    s.server = ServerConfig(
        host=host, port=8890, auth_token=token,
        require_auth_off_loopback=strict,
    )
    return s


# -- is_loopback_host -------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.5.4.3"])
def test_loopback_hosts(host):
    assert is_loopback_host(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "10.0.0.1", "example.com", ""])
def test_non_loopback_hosts(host):
    assert is_loopback_host(host) is False


# -- enforce_bind_safety ----------------------------------------------------


def test_loopback_no_token_is_fine():
    assert enforce_bind_safety(_settings("127.0.0.1")) is None


def test_non_loopback_no_token_refused():
    with pytest.raises(RuntimeError, match="Refusing to bind"):
        enforce_bind_safety(_settings("0.0.0.0"))


def test_non_loopback_with_token_allowed():
    assert enforce_bind_safety(_settings("0.0.0.0", token="s3cret")) is None


def test_non_loopback_opt_out_warns():
    warn = enforce_bind_safety(_settings("0.0.0.0", strict=False))
    assert warn is not None and "WARNING" in warn


# -- RequireBearer middleware -----------------------------------------------


class _App:
    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


class _Messages:
    """Collect what the middleware sent without a real ASGI server."""

    def __init__(self):
        self.sent: list[dict] = []

    async def __call__(self, message):
        self.sent.append(message)


async def _run(mw, headers):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
    }
    msgs = _Messages()
    await mw(scope, None, msgs)
    return msgs.sent


def test_bearer_correct_token_passes():
    asyncio.run(_bearer_correct())


async def _bearer_correct():
    app = _App()
    mw = RequireBearer(app, "tok")
    sent = await _run(mw, [("Authorization", "Bearer tok")])
    assert app.calls == 1
    assert sent[0]["status"] == 200


def test_bearer_wrong_token_401():
    asyncio.run(_bearer_wrong())


async def _bearer_wrong():
    app = _App()
    mw = RequireBearer(app, "tok")
    sent = await _run(mw, [("Authorization", "Bearer wrong")])
    assert app.calls == 0
    assert sent[0]["status"] == 401
    assert b"unauthorized" in sent[1]["body"]


def test_bearer_missing_header_401():
    asyncio.run(_bearer_missing())


async def _bearer_missing():
    app = _App()
    mw = RequireBearer(app, "tok")
    sent = await _run(mw, [])
    assert app.calls == 0
    assert sent[0]["status"] == 401


def test_bearer_non_http_scope_passes_through():
    asyncio.run(_bearer_non_http())


async def _bearer_non_http():
    app = _App()
    mw = RequireBearer(app, "tok")
    msgs = _Messages()
    await mw({"type": "lifespan"}, None, msgs)
    assert app.calls == 1


def test_wrap_http_app_noop_without_token():
    app = object()
    assert wrap_http_app(app, _settings("127.0.0.1")) is app


def test_wrap_http_app_wraps_with_token():
    app = object()
    wrapped = wrap_http_app(app, _settings("127.0.0.1", token="t"))
    assert isinstance(wrapped, RequireBearer)
