# F1 Live Dashboard

Real-time Formula 1 race dashboard: a Python backend turns the F1 live-timing feed into a clean
WebSocket stream, and a dependency-free browser frontend (vanilla ES modules, canvas) renders it.

    Source (live | file | sim) -> Msg -> RaceEngine -> WebSocket JSON -> browser

The wire formats are specified in [`docs/CONTRACT.md`](docs/CONTRACT.md).

## Setup

    python -m venv .venv && source .venv/bin/activate     # Python 3.11+ (tested on 3.13)
    pip install -r requirements.txt

## Run

    python -m backend --mode sim  --speed 5 --laps 20      # synthetic race, no network needed
    python -m backend --mode file --file recordings/xyz.txt --speed 2   # replay a recording
    python -m backend --mode live                          # real session (see below)

Then open <http://localhost:8000>. Options: `--port 8000 --host 0.0.0.0 --seed N --no-auth --log-level debug`.

Useful endpoints: `/api/state`, `/api/history`, `/api/track`, `/api/status` (mode, clients, msgs/sec,
last message age, engine counts), `POST /api/control {speed, paused, seek}`, WebSocket `/ws`.

## Race weekend (live mode)

1. Start the server about 15 minutes before the session: `python -m backend --mode live`.
   It connects to the F1 SignalR feed through `fastf1.livetiming`, tails the stream and tees the raw
   feed to `recordings/<timestamp>.txt` so you can replay the session afterwards.
2. Live timing needs an F1 TV subscription token. Authenticate with fastf1 first
   (`python -m fastf1.livetiming ...` / see the fastf1 docs on live timing authentication); if the
   token is missing or expired the connection is refused and `/api/status` shows the error.
   `--no-auth` tries the unauthenticated endpoint, which F1 only serves intermittently.
3. The track outline is learned from car position data after the first clean lap, so the map
   appears a couple of minutes into a session. It is kept for the rest of the run.
4. If the feed drops the server keeps serving the last state and retries with backoff.

## Testing without a race

* `--mode sim` generates a synthetic race (pit stops, safety cars, radio, penalties) with the same
  topics as the real feed. `--speed 20 --laps 8` finishes in about a minute.
* `--mode file` replays any recording at `--speed N`; pause, change speed and seek from the UI
  (or `POST /api/control`). Seeking resets the engine and re-feeds from the start.

## Tests

    python -m pytest backend/tests -q

## Layout

    backend/ingest   raw feed parsing + sources (live, file, sim)
    backend/sim      synthetic race generator
    backend/engine   RaceEngine: raw patches -> state / fast / history
    backend/track.py track outline builder + car projection
    backend/server.py FastAPI app, WebSocket broadcast
    frontend/        static dashboard served at /
