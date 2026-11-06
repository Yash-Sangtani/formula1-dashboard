"""python -m backend doctor: checks with faked environment/network (nothing here touches the real internet)."""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from backend import doctor as D
from backend.doctor import Deps, Result


class FakeDeps(Deps):
    def __init__(self, **kw):
        super().__init__(env=kw.pop("env", {}))
        self.versions = kw.pop("versions", {"fastf1": "3.9.2", "fastapi": "0.1", "uvicorn": "0.2", "numpy": "2.0"})
        self.modules = kw.pop("modules", {"websockets", "signalrcore", "requests", "jwt"})
        self.py = kw.pop("py", (3, 13, 1))
        self.free = kw.pop("free", 50_000)
        self.in_use = kw.pop("in_use", False)
        self.dns = kw.pop("dns", ["1.2.3.4"])
        self.neg = kw.pop("neg", (200, {"AWSALBCORS": "x"}))
        self.token = kw.pop("token", {"none": True, "method": "password"})
        self.login_result = kw.pop("login_result", None)
        self.probe_result = kw.pop("probe_result", None)
        self.calls = []

    def module_version(self, name):
        return self.versions.get(name)

    def has_module(self, name):
        return name in self.modules

    def python_version(self):
        return self.py

    def disk_free_mb(self, path):
        return self.free

    def port_in_use(self, host, port):
        return self.in_use

    def resolve(self, host):
        self.calls.append("dns")
        if isinstance(self.dns, Exception):
            raise self.dns
        return self.dns

    def negotiate(self, timeout=10.0):
        self.calls.append("negotiate")
        if isinstance(self.neg, Exception):
            raise self.neg
        return self.neg

    def token_state(self):
        return self.token

    def login(self):
        self.calls.append("login")
        return self.login_result

    def probe(self, seconds, no_auth):
        self.calls.append("probe")
        if isinstance(self.probe_result, Exception):
            raise self.probe_result
        return self.probe_result


def by_name(results, prefix):
    return [r for r in results if r.name.startswith(prefix)]


def one(results, prefix):
    r = by_name(results, prefix)
    assert len(r) == 1, [x.name for x in results]
    return r[0]


def opts(tmp_path, **kw):
    ns = dict(selftest=False, probe_seconds=3.0, skip_network=False, no_login=False, no_auth=False,
              recordings_dir=str(tmp_path / "rec"), port=8000, host="0.0.0.0")
    ns.update(kw)
    return argparse.Namespace(**ns)


GOOD_TOKEN = {"valid": True, "expires_at": time.time() + 72 * 3600, "status": "active", "product": "F1 TV Pro", "method": "password"}


# ----------------------------------------------------------------------------- environment
def test_python_and_packages():
    assert D.check_python(FakeDeps(py=(3, 13, 1)))[0].status == "PASS"
    assert D.check_python(FakeDeps(py=(3, 10, 4)))[0].status == "WARN"
    assert D.check_python(FakeDeps(py=(3, 9, 9)))[0].status == "FAIL"
    ok = D.check_packages(FakeDeps())
    assert all(r.status == "PASS" for r in ok), [r.line() for r in ok]
    bad = D.check_packages(FakeDeps(versions={"fastapi": "1", "uvicorn": "1", "numpy": "1"}, modules=set()))
    assert one(bad, "fastf1").status == "FAIL" and "pip install" in one(bad, "fastf1").hint
    assert one(bad, "websocket library").status == "FAIL"
    assert one(D.check_packages(FakeDeps(versions={"fastf1": "3.4.2", "fastapi": "1", "uvicorn": "1", "numpy": "1"})), "fastf1").status == "FAIL"
    assert one(D.check_packages(FakeDeps(versions={"fastf1": "3.12.0", "fastapi": "1", "uvicorn": "1", "numpy": "1"})), "fastf1").status == "WARN"
    deps_missing = D.check_packages(FakeDeps(modules={"websockets"}))
    assert one(deps_missing, "fastf1 dependencies").status == "FAIL"


def test_recordings_dir_and_disk(tmp_path):
    r = D.check_recordings(FakeDeps(), str(tmp_path / "new" / "rec"))
    assert [x.status for x in r] == ["PASS", "PASS"] and (tmp_path / "new" / "rec").is_dir()
    assert list((tmp_path / "new" / "rec").iterdir()) == []  # probe file removed
    f = tmp_path / "afile"
    f.write_text("x")
    r = D.check_recordings(FakeDeps(), str(f / "sub"))
    assert r[0].status == "FAIL" and "--recordings-dir" in r[0].hint
    assert one(D.check_recordings(FakeDeps(free=500), str(tmp_path / "a")), "disk").status == "WARN"
    assert one(D.check_recordings(FakeDeps(free=50), str(tmp_path / "b")), "disk").status == "FAIL"


def test_port_check():
    assert D.check_port(FakeDeps(), "127.0.0.1", 8000)[0].status == "PASS"
    r = D.check_port(FakeDeps(in_use=True), "127.0.0.1", 8000)[0]
    assert r.status == "FAIL" and "--port" in r.hint


def test_port_check_real_socket():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert Deps().port_in_use("127.0.0.1", port) is True
    finally:
        s.close()
    assert Deps().port_in_use("127.0.0.1", port) is False


# ----------------------------------------------------------------------------- credentials (never print secrets)
def test_credentials_valid_token_reports_expiry_not_secrets():
    d = FakeDeps(token=GOOD_TOKEN, env={"F1_USERNAME": "me@example.com", "F1_PASSWORD": "hunter2"})
    r = D.check_credentials(d, no_auth=False)
    assert r[0].status == "PASS" and "active" in r[0].detail
    text = "\n".join(x.line() for x in r)
    assert "hunter2" not in text and "me@example.com" not in text
    assert "login" not in d.calls


def test_credentials_inactive_subscription_and_expiring_token_warn():
    t = dict(GOOD_TOKEN, status="inactive")
    assert D.check_credentials(FakeDeps(token=t), False)[0].status == "WARN"
    t = dict(GOOD_TOKEN, expires_at=time.time() + 3600)
    assert D.check_credentials(FakeDeps(token=t), False)[0].status == "WARN"


def test_credentials_missing_is_fail_with_both_fixes():
    r = D.check_credentials(FakeDeps(), False)
    f = [x for x in r if x.status == "FAIL"][0]
    assert "F1_USERNAME" in f.hint and "--authenticate" in f.hint and "--no-auth" in f.hint


def test_credentials_env_triggers_login_check_and_reports_failure():
    env = {"F1_USERNAME": "u", "F1_PASSWORD": "secret-pw"}
    d = FakeDeps(env=env, login_result=None)
    r = D.check_credentials(d, False)
    assert one(r, "F1 sign-in").status == "PASS" and d.calls == ["login"]
    d = FakeDeps(env=env, login_result="F1AuthError: Sign-in failed: bad credentials")
    r = D.check_credentials(d, False)
    assert one(r, "F1 sign-in").status == "FAIL" and "secret-pw" not in "\n".join(x.line() for x in r)
    d = FakeDeps(env=env)
    r = D.check_credentials(d, False, login=False)
    assert one(r, "F1 sign-in").status == "INFO" and d.calls == []
    # expired stored token + env creds: renew via login
    d = FakeDeps(env=env, token={"valid": False, "expires_at": 1.0, "method": "password"})
    assert one(D.check_credentials(d, False), "F1 sign-in").status == "PASS"


def test_credentials_no_auth_is_info():
    assert D.check_credentials(FakeDeps(), True)[0].status == "INFO"


# ----------------------------------------------------------------------------- network
def test_network_all_good():
    r, ok = D.check_network(FakeDeps())
    assert ok and [x.status for x in r] == ["PASS", "PASS"]


def test_network_dns_failure_stops_early_with_hint():
    d = FakeDeps(dns=OSError("Name or service not known"))
    r, ok = D.check_network(d)
    assert not ok and r[-1].status == "FAIL" and "DNS" in r[-1].name and d.calls == ["dns"]


def test_network_dns_failure_behind_proxy_is_only_a_warning():
    d = FakeDeps(dns=OSError("nope"), env={"HTTPS_PROXY": "http://p:3128"})
    r, ok = D.check_network(d)
    assert ok and "negotiate" in d.calls
    assert one(r, "proxy").status == "WARN" and one(r, "DNS").status == "WARN"


@pytest.mark.parametrize("exc,word", [
    (type("SSLError", (Exception,), {})("certificate verify failed"), "REQUESTS_CA_BUNDLE"),
    (type("ProxyError", (Exception,), {})("Tunnel connection failed: 403 Forbidden"), "HTTPS_PROXY"),
    (type("ReadTimeout", (Exception,), {})("timed out"), "timed out"),
    (ConnectionError("refused"), "firewall"),
])
def test_network_negotiate_failures_have_specific_hints(exc, word):
    r, ok = D.check_network(FakeDeps(neg=exc))
    assert not ok and r[-1].status == "FAIL" and word in r[-1].hint


def test_network_missing_awsalbcors_cookie_fails():
    r, ok = D.check_network(FakeDeps(neg=(200, {})))
    assert not ok and r[-1].status == "FAIL" and "AWSALBCORS" in r[-1].detail


# ----------------------------------------------------------------------------- probe interpretation
def probe(**kw):
    base = {"topics": {}, "connected": False, "error": None, "info": {}}
    base.update(kw)
    return D.check_probe(FakeDeps(probe_result=base), 15, False)[0]


def test_probe_variants():
    r = probe(connected=False, error="Cannot reach livetiming.formula1.com (network unreachable)")
    assert r.status == "FAIL" and "network" in r.hint
    r = probe(connected=False, error="F1 sign-in failed (F1AuthError: bad)")
    assert r.status == "FAIL" and "credentials" in r.hint
    assert probe(connected=False).status == "WARN"
    r = probe(connected=True)
    assert r.status == "PASS" and "no session is running" in r.detail
    r = probe(connected=True, topics={"SessionInfo": 1, "SessionStatus": 1, "Heartbeat": 3},
              info={"sessionStatus": "Inactive", "session": {"meeting": "Test GP", "name": "Practice 1"}})
    assert r.status == "PASS" and "no session is running" in r.detail and "Practice 1" in r.detail
    r = probe(connected=True, topics={"CarData": 50, "Position": 40, "TimingData": 10},
              info={"sessionStatus": "Started", "session": {"name": "Race"}})
    assert r.status == "PASS" and "RECEIVING" in r.detail and "CarData" in r.detail
    r = D.check_probe(FakeDeps(probe_result=RuntimeError("boom")), 5, False)[0]
    assert r.status == "FAIL" and "boom" in r.detail


def test_run_checks_skips_probe_when_network_fails_and_honours_flags(tmp_path):
    d = FakeDeps(dns=OSError("x"), token=GOOD_TOKEN)
    res = D.run_checks(opts(tmp_path), d)
    assert "probe" not in d.calls and one(res, "live probe").status == "WARN"
    d = FakeDeps(token=GOOD_TOKEN, probe_result={"topics": {}, "connected": True, "error": None, "info": {}})
    res = D.run_checks(opts(tmp_path), d)
    assert d.calls == ["dns", "negotiate", "probe"]
    d = FakeDeps(token=GOOD_TOKEN)
    D.run_checks(opts(tmp_path, skip_network=True), d)
    assert d.calls == []
    d = FakeDeps(token=GOOD_TOKEN)
    res = D.run_checks(opts(tmp_path, probe_seconds=0), d)
    assert "probe" not in d.calls and one(res, "live probe").status == "INFO"


def test_selftest_mode_runs_only_offline_checks(tmp_path):
    d = FakeDeps()
    res = D.run_checks(opts(tmp_path, selftest=True), d, selftest=lambda: [Result("PASS", "selftest x")])
    assert d.calls == [] and one(res, "selftest x")
    assert not by_name(res, "port") and not by_name(res, "F1")


# ----------------------------------------------------------------------------- main(): output and exit codes
def run_main(argv, deps):
    buf = io.StringIO()
    code = D.main(argv, deps=deps, out=buf)
    return code, buf.getvalue()


def test_main_exit_codes_and_output(tmp_path):
    good = FakeDeps(token=GOOD_TOKEN, probe_result={"topics": {}, "connected": True, "error": None, "info": {}})
    code, out = run_main(["--recordings-dir", str(tmp_path / "r"), "--probe-seconds", "2", "--port", "8123"], good)
    assert code == 0 and "[PASS] fastf1" in out and "READY" in out and "NOT READY" not in out
    bad = FakeDeps(token=GOOD_TOKEN, in_use=True)
    code, out = run_main(["--recordings-dir", str(tmp_path / "r"), "--skip-network"], bad)
    assert code == 1 and "[FAIL] port 8000" in out and "fix:" in out and "NOT READY" in out
    warn = FakeDeps(token=dict(GOOD_TOKEN, status="inactive"))
    code, out = run_main(["--recordings-dir", str(tmp_path / "r"), "--skip-network"], warn)
    assert code == 0 and "[WARN]" in out and "with warnings" in out


# ----------------------------------------------------------------------------- the real probe, through a fake hub
def test_default_probe_connects_via_real_fastf1_client_and_reports_topics(tmp_path, monkeypatch):
    pytest.importorskip("fastf1")
    import backend.tests.test_ingest_sources as T

    conn = T._FakeConn(feed=[["Heartbeat", {"Utc": "x"}, "2025-03-16T04:00:00.123Z"],
                             ["SessionStatus", {"Status": "Started"}, "2025-03-16T04:00:01.000Z"],
                             ["TimingData", {"Lines": {}}, "2025-03-16T04:00:02.000Z"]])
    T._patch_hub(monkeypatch, [conn])
    r = Deps().probe(3.0, no_auth=True)
    assert r["connected"] is True and r["error"] is None
    assert r["topics"].get("TimingData") == 1 and r["topics"].get("SessionStatus", 0) >= 1
    res = D.check_probe(FakeDeps(probe_result=r), 3, True)[0]
    assert res.status == "PASS" and "RECEIVING" in res.detail
    assert conn.stopped


def test_default_probe_reports_blocked_network(tmp_path, monkeypatch):
    pytest.importorskip("fastf1")
    import fastf1.livetiming.client as fc

    def boom(*a, **k):
        raise ConnectionError("HTTPSConnectionPool(host='livetiming.formula1.com'): Max retries exceeded")

    monkeypatch.setattr(fc.requests, "options", boom)
    r = Deps().probe(3.0, no_auth=True)
    assert r["connected"] is False and "Cannot reach livetiming.formula1.com" in r["error"]
    assert D.check_probe(FakeDeps(probe_result=r), 3, True)[0].status == "FAIL"


# ----------------------------------------------------------------------------- selftest + CLI wiring
def test_selftest_passes_end_to_end():
    res = D.run_selftest()
    assert res and all(r.status in ("PASS", "WARN") for r in res), [r.line() for r in res]
    assert any(r.name.startswith("selftest sim") and r.status == "PASS" for r in res)
    assert any(r.name.startswith("selftest live pipeline") and r.status == "PASS" for r in res)


def test_cli_subcommand_wired_without_breaking_flags():
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m", "backend", "doctor", "--selftest"], capture_output=True, text=True,
                       timeout=180, cwd=str(root))
    assert r.returncode == 0 and "READY" in r.stdout, r.stdout + r.stderr
    h = subprocess.run([sys.executable, "-m", "backend", "--help"], capture_output=True, text=True, timeout=60, cwd=str(root))
    assert h.returncode == 0 and "--recordings-dir" in h.stdout and "--mode" in h.stdout and "--no-auth" in h.stdout


def test_main_refuses_busy_port():
    import socket

    root = Path(__file__).resolve().parents[2]
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        r = subprocess.run([sys.executable, "-m", "backend", "--mode", "sim", "--host", "127.0.0.1", "--port", str(port)],
                           capture_output=True, text=True, timeout=60, cwd=str(root))
    finally:
        s.close()
    assert r.returncode != 0 and "already in use" in (r.stderr + r.stdout)
