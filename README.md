# F1 Live Dashboard

Real-time Formula 1 race dashboard: a Python backend turns the F1 live-timing feed into a clean
WebSocket stream, and a dependency-free browser frontend (vanilla ES modules, canvas) renders it.

    Source (live | file | sim) -> Msg -> RaceEngine -> WebSocket JSON -> browser

The wire formats are specified in [`docs/CONTRACT.md`](docs/CONTRACT.md).

## Setup

    python -m venv .venv && source .venv/bin/activate     # Python 3.11+ (tested on 3.13)
    pip install -r requirements.txt          # add requirements-dev.txt for tests

## Run

    python -m backend --mode sim  --speed 5 --laps 20      # synthetic race, no network needed
    python -m backend --mode file --file recordings/xyz.txt --speed 2   # replay a recording
    python -m backend --mode live                          # real session (see below)

Then open <http://localhost:8000>. Options: `--port 8000 --host 0.0.0.0 --seed N --no-auth --recordings-dir DIR --max-clients N --log-level debug`.

Useful endpoints: `/api/state`, `/api/history`, `/api/track`, `/api/status` (mode, phase, clients, msgs/sec,
last message age, recording, engine counts), `POST /api/control {speed, paused, seek}`, `POST /api/track/reset`, WebSocket `/ws`.
`python -m backend doctor [--selftest]` runs the pre-flight checks.

## Race weekend (live mode)

**Honest status:** the live path (fastf1 client -> recording -> tail -> engine -> browser) is covered by unit tests with fakes and
by driving fastf1's real client class through a fake SignalR hub, but it has **never been run against the real F1 feed** (the
build sandbox cannot reach `livetiming.formula1.com`). Expect to watch the first session and keep the doctor output handy.
Practice/qualifying/sprint handling is written from recalled feed shapes, not from real recordings.

### Checklist (do this once, the evening before)

1. `pip install -r requirements.txt`, then `python -m backend doctor --selftest` - offline; must end with `READY`.
2. Credentials: `export F1_USERNAME=... F1_PASSWORD=...` (needs an F1 TV subscription) **or** run
   `python -m fastf1 auth f1tv --authenticate` once. Do not paste them anywhere else; nothing prints them.
3. `python -m backend doctor` - checks versions, recordings folder + disk space, port 8000, token/sign-in, DNS + HTTPS
   (`/signalrcore/negotiate` must return the `AWSALBCORS` cookie) and then opens a real live connection for 15 s.
   Every `[FAIL]` has a `fix:` line; warnings do not block. Outside a session the probe reports "connected, no session
   running" - that is a pass. Re-run it on the actual machine/network you will use (a hotel/corporate proxy may allow
   HTTPS but block websockets).

### Session day

4. Start the server **before** the session, any time - 30 minutes early is fine: `python -m backend --mode live`
   (add `--recordings-dir D:/f1rec` to choose where the raw feed is stored, `--port N` if 8000 is taken). It refuses to
   start if the port is busy. Open <http://localhost:8000>.
5. **"Waiting for session" is normal.** It means: signed in, websocket connected, the feed has nothing live yet
   (`phase: "waiting"` in `/api/status`, the `hello` frame and a `status` WS frame). The server never gives up: it keeps the
   connection, re-connects with back-off if it drops (recycles an idle socket every ~5 min before a session), and
   switches to `live` by itself when `SessionStatus: Started` arrives. You do not need to restart anything between
   Practice, Qualifying, Sprint and Race: leave it running (see "restart or not" below).
6. `phase: "error"` shows the reason in `/api/status` -> `error` (and in the console). Typical: *sign-in failed* (fix
   credentials; the server retries every ~30 s, no restart needed once the token appears), *cannot reach
   livetiming.formula1.com* (network), *rejected the subscription* (subscription inactive). Run
   `python -m backend doctor` in a second terminal while the server runs (use `--port 8001` to skip the port check).
7. The track outline: for a new circuit the server first asks the MultiViewer circuit API for the official layout
   (5 s timeout, silent on failure) and only adopts it if the cars' positions actually lie on it; otherwise it learns the
   outline from the first clean laps (a few minutes into practice). Wrong or odd outline? `curl -X POST
   localhost:8000/api/track/reset` relearns it. A new meeting (new circuit) resets it automatically.
8. Quick health glance during a session: `curl localhost:8000/api/status` -> `phase`, `lastMessageAgeSec` (a few seconds
   when live), `recording.bytes` growing, `engine.errors` ~0, `error` null.

**Restart or not between sessions?** Leaving it running is supported: the recording is *one file per server start*, so
several sessions end up back-to-back in it. If you want one clean file per session (easier to replay), stop the server
(Ctrl+C) after each session and start it again; it only costs the learned track, which comes back from the official layout.

### Afterwards

* Recordings are plain text files `recordings/YYYYMMDD_HHMMSS.txt` (the raw fastf1 format, UTF-8). Replay one:
  `python -m backend --mode file --file recordings/20250316_040000.txt --speed 4` (pause/seek in the UI).
  Reconnects append further snapshot lines to the same file; that is expected.
* Size: the file grows continuously while the feed is up (telemetry + positions at several Hz). Rough expectation is tens
  to a couple of hundred MB per session - **not measured**; the doctor warns below 2 GB free and `/api/status`
  shows `recording.bytes` and `diskFreeMB`.
* Lines that cannot be parsed are skipped and counted (`control.badLines`).

### Known limits / untested

* Never run against the real feed; the first minutes of Practice 1 are the real test. If the dashboard stays on
  "waiting" while the F1 app shows a live session, check `/api/status` -> `control` (`connected`, `lastFeedAgeSec`,
  `authState`) and the `recording.bytes` counter before suspecting the UI.
* fastf1's own client has no auto-reconnect and can hang while connecting; this project supervises it (watchdog, recycle on
  silence/close, never gives up) - that logic is tested with fakes only.
* A token that expires mid-session does not drop the already open connection; the next reconnect renews it
  (needs `F1_USERNAME`/`F1_PASSWORD` or a fresh `fastf1 auth`).
* Hosting from a datacentre IP may be refused by F1; running at home is the safest.

## Testing without a race

* `--mode sim` generates a synthetic race (pit stops, safety cars, radio, penalties) with the same
  topics as the real feed. `--speed 20 --laps 8` finishes in about a minute.
* `--mode file` replays any recording at `--speed N`; pause, change speed and seek from the UI
  (or `POST /api/control`). Seeking resets the engine and re-feeds from the start.

## 3D view (key 5)

The **3D** tab renders the circuit as a 3D ribbon (elevation is learned from live position data)
with a chase camera on the selected car (or the leader). It is an approximate chase/onboard-style
view built from position telemetry, not a true cockpit camera.

## Docker / hosting

    docker compose up --build                                          # sim mode on :8000
    F1_MODE=live DASHBOARD_TOKEN=pick-a-secret F1_USERNAME=... F1_PASSWORD=... docker compose up -d --build

Configuration is also available as env vars: `F1_MODE`, `F1_FILE`, `F1_SPEED`, `F1_NO_AUTH`, `PORT`,
`HOST`, `LOG_LEVEL`, `DASHBOARD_TOKEN`, `F1_RECORDINGS_DIR`, `F1_MAX_CLIENTS`, `F1_NO_TRACK_FALLBACK` (1 = never contact the circuit-layout API). See [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Tests

    python -m pytest -q

## Layout

    backend/ingest   raw feed parsing + sources (live, file, sim)
    backend/sim      synthetic race generator
    backend/engine   RaceEngine: raw patches -> state / fast / history
    backend/track.py track outline builder + official-layout fallback + car projection
    backend/doctor.py pre-flight checks (`python -m backend doctor`)
    backend/server.py FastAPI app, WebSocket broadcast
    frontend/        static dashboard served at /
