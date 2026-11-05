from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from backend.auth import TokenMiddleware
from backend.ingest.sources import SimSource
from backend.server import create_app


def make():
    app = create_app(SimSource(seed=1, speed=20, laps=2), mode="sim")
    app.add_middleware(TokenMiddleware, token="s3cret")
    return app


def test_http_requires_token_and_sets_cookie():
    with TestClient(make()) as c:
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status", headers={"x-token": "nope"}).status_code == 401
        assert c.get("/healthz").status_code == 200
        r = c.get("/api/status?token=s3cret")
        assert r.status_code == 200 and "f1token=s3cret" in r.headers["set-cookie"]
        assert c.get("/api/status").status_code == 200  # cookie persisted


def test_websocket_requires_token():
    with TestClient(make()) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws"):
                pass
        with c.websocket_connect("/ws?token=s3cret") as ws:
            assert ws.receive_json()["type"] == "hello"
