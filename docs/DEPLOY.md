# Hosting the dashboard

The server is one Python process (FastAPI + WebSocket) that also serves the static frontend. It holds
the live race state in memory, so run **exactly one instance** and keep it always-on during sessions
(no scale-to-zero, no multiple replicas behind a load balancer).

## 1. Locally / on your own machine
    pip install -r requirements.txt
    python -m backend --mode live        # export F1_USERNAME / F1_PASSWORD first
Open http://localhost:8000 (or `http://<your-ip>:8000` from a phone on the same network).

## 2. Docker
    docker compose up -d --build
Settings come from env vars (see `docker-compose.yml`). Raw recordings of live sessions are written to the
`recordings` volume; replay one later with `F1_MODE=file F1_FILE=/app/recordings/<file>.txt`.

## 3. A VPS or any container host
Any host that runs a long-lived container with WebSocket support works (Fly.io, Railway, Render *paid/always-on*
plans, a small VPS). Checklist:
* One instance, always on; port `8000` (or `PORT`); health check path `/healthz`.
* **Set `DASHBOARD_TOKEN`.** The live feed runs on *your* F1 TV account, so don't expose it openly. Open the
  site once with `https://your-host/?token=<token>`; a cookie keeps you signed in (phones too).
* Terminate TLS in front (the platform's edge, or Caddy: `your.domain { reverse_proxy localhost:8000 }`).
  The browser then uses `wss://` automatically.
* Put `F1_USERNAME` / `F1_PASSWORD` in the platform's secret store, not in the image or repo.
* Mount a persistent volume at `/app/recordings` if you want to keep session recordings.
* F1's live-timing feed is only available while a session is running; outside sessions use `F1_MODE=sim`
  or a recording to demo.

## Caveats
* F1 may block some datacentre IP ranges or change the unofficial feed without notice. If hosted live mode is
  refused, run locally on a home connection.
* This has not been tested against a real live session or a real cloud host from the build sandbox.
