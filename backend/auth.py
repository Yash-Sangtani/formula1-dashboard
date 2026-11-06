"""Optional shared-token gate for hosted deployments.

When DASHBOARD_TOKEN is set, every HTTP and WebSocket request must carry the token as
`?token=...`, an `x-token` header, or the `f1token` cookie. A valid `?token=` on a page
request sets the cookie so the browser (and its WebSocket) stay authorised.
`/healthz` is always open so platform health checks work.
"""
from __future__ import annotations

import hmac
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

OPEN_PATHS = {"/healthz"}
COOKIE = "f1token"


def _ok(given: str | None, token: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), token.encode())


class TokenMiddleware:
    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("path") in OPEN_PATHS:
            return await self.app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        query = parse_qs(scope.get("query_string", b"").decode())
        q_tok = (query.get("token") or [None])[0]
        cookie = SimpleCookie(headers.get("cookie", "")).get(COOKIE)
        given = q_tok or headers.get("x-token") or (cookie.value if cookie else None)

        if _ok(given, self.token):
            if scope["type"] == "http" and q_tok:
                async def send_with_cookie(msg):
                    if msg["type"] == "http.response.start":
                        msg = dict(msg)
                        msg["headers"] = list(msg.get("headers", [])) + [
                            (b"set-cookie", f"{COOKIE}={q_tok}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000".encode())
                        ]
                    await send(msg)
                return await self.app(scope, receive, send_with_cookie)
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            await receive()  # consume websocket.connect
            return await send({"type": "websocket.close", "code": 4401})
        body = b"Unauthorized: open the dashboard with ?token=<DASHBOARD_TOKEN>"
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
