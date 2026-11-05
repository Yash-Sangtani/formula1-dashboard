# F1 Live Dashboard — shared contract

Pipeline:  Source (live | file | sim)  →  `Msg`  →  `RaceEngine`  →  WebSocket JSON  →  browser.

## 1. `Msg` (backend/ingest/messages.py)
```python
@dataclass
class Msg:
    topic: str      # F1 SignalR topic, ".z" suffix already stripped & payload decoded (base64+raw-deflate+json)
    data: Any       # decoded JSON (dict/list). Patches, NOT snapshots: engine must deep-merge.
    utc: float      # unix seconds (float) of the message (from the feed timestamp)
```
Raw recorded-file line format (what fastf1 SignalRClient writes), one per line, Python-repr of a list:
`['TimingData', {...}, '2024-03-02T15:04:05.123Z']`  (`.z` topics: `['CarData.z', '<base64 str>', '<ts>']`).
Also accept JSON-lines `{"topic":..., "data":..., "utc":...}`. `parse_line(line)->Msg|None`.

Sources (backend/ingest/sources.py) — all implement `async def messages(self) -> AsyncIterator[Msg]`
and a `.control` object with `speed: float`, `paused: bool`, `async seek(utc)`, `info()`:
* `LiveSource` — wraps `fastf1.livetiming.client.SignalRClient` (writes to a file; we tail it). Needs F1 TV auth (`fastf1.livetiming` token) or `no_auth=True`; also tees raw feed to `recordings/<ts>.txt`.
* `FileSource(path, speed)` — replays a recording honoring original timestamp spacing × speed.
* `SimSource(seed, speed, laps)` — synthetic race (backend/sim) emitting the same topics/shape.

## 2. Raw topics the engine must handle (patch semantics: dicts deep-merge; list-valued `Messages`/`Captures` may arrive as list OR as dict keyed by index)
* `DriverList` `{ "1": {RacingNumber,Tla,FirstName,LastName,TeamName,TeamColour("3671C6"),Line} }`
* `TimingData` `{Lines:{ "1": {Position:"1", GapToLeader:"+1.234", IntervalToPositionAhead:{Value:"+0.512"},
  LastLapTime:{Value:"1:32.123",OverallFastest,PersonalFastest}, BestLapTime:{Value}, 
  Sectors:{"0":{Value:"28.1",OverallFastest,PersonalFastest}, "1":..,"2":..}, NumberOfLaps:5, NumberOfPitStops:1,
  InPit,PitOut,Retired,Stopped, Speeds:{I1:{Value:"310"},I2,FL,ST}, KnockedOut, Status:int }}}` (gap may be "1L"/"LAP 2"/"" for leader)
* `TimingAppData` `{Lines:{"1":{Stints:{"0":{Compound:"SOFT",New:"true",TotalLaps:3,StartLaps:0}}, GridPos:"3"}}}` (Stints may be list or dict)
* `CarData.z` decoded: `{Entries:[{Utc:"..Z",Cars:{"1":{Channels:{"0":rpm,"2":speed,"3":gear,"4":throttle,"5":brake(0/100 or 0/1),"45":drs}}}}]}`  (drs>=10 and in (10,12,14) = open)
* `Position.z` decoded: `{Position:[{Timestamp:"..Z",Entries:{"1":{Status:"OnTrack",X:1234,Y:-567,Z:10}}}]}`
* `RaceControlMessages` `{Messages:[{Utc,Lap,Category("Flag"|"Other"|"SafetyCar"|"Drs"),Flag,Scope,Sector,Message,RacingNumber?}]}`
* `TrackStatus` `{Status:"1"|"2"|"4"|"5"|"6"|"7",Message}`  1 clear,2 yellow,4 SC,5 red,6 VSC,7 VSC ending
* `WeatherData` `{AirTemp,Humidity,Pressure,Rainfall,TrackTemp,WindDirection,WindSpeed}` (strings)
* `LapCount` `{CurrentLap,TotalLaps}`   * `SessionStatus` `{Status:"Inactive|Started|Aborted|Finished|Finalised|Ends"}`
* `SessionInfo` `{Meeting:{Name,Location,Country:{Name}},Name,Type,Path}`   * `ExtrapolatedClock` `{Utc,Remaining:"01:23:45",Extrapolating:bool}`
* `TeamRadio` `{Captures:[{Utc,RacingNumber,Path}]}`   * `Heartbeat` ignore.

## 3. WebSocket `/ws` (server→client, JSON). Car keys are racing-number STRINGS ("1","44").
Client may send `{"type":"control","speed":5}`, `{"type":"control","paused":true}`, `{"type":"control","seek":<utc>}` (replay only).

### `{"type":"hello","mode":"live|file|sim","replay":bool,"info":{...}}` — once on connect.
### `{"type":"state", ...}` — on connect, then ≤1 Hz when changed:
```jsonc
{ "type":"state", "utc":1700000000.0,
  "session": {"name":"Race","meeting":"Bahrain Grand Prix","type":"Race","status":"Started"},
  "clock":   {"remainingSec":5400.0|null,"extrapolating":true,"elapsedSec":123.4|null},
  "lap":     {"current":12,"total":57|null},
  "trackStatus":{"code":"1","label":"AllClear","flag":"green|yellow|sc|vsc|red"},
  "weather": {"air":28.1,"track":35.0,"humidity":40,"pressure":1010,"wind":2.3,"windDir":120,"rain":false},
  "order": ["1","44",...],                         // by position
  "cars": { "1": {
      "num":"1","tla":"VER","name":"Max Verstappen","team":"Red Bull Racing","color":"#3671C6",
      "pos":1,"grid":1,"posChange":0,              // posChange = grid - pos
      "gapLeader":"+0.000","interval":"+0.512","gapLeaderSec":0.0|null,"intervalSec":0.512|null,
      "lap":12,"pits":1,"inPit":false,"pitOut":false,"retired":false,"stopped":false,
      "last":{"text":"1:32.123","sec":92.123|null,"overall":false,"personal":true},
      "best":{"text":"1:31.9","sec":91.9,"overall":true,"personal":true},
      "sectors":[{"text":"28.1","sec":28.1,"overall":false,"personal":true}, {..},{..}],   // latest completed value per sector; missing => nulls
      "speedTraps":{"i1":310,"i2":305,"fl":300,"st":322},
      "tyre":{"compound":"SOFT","age":5,"new":true,"stint":2},
      "stints":[{"compound":"MEDIUM","new":true,"startLap":0,"laps":14},{...}],
      "tel": {"speed":312,"rpm":11500,"gear":7,"throttle":100,"brake":0,"drs":true,"drsRaw":12},   // last known
      "xy":  {"x":1234,"y":-567,"z":10,"status":"OnTrack"} | null,
      "progress": 0.4321|null                       // fraction of lap distance along track polyline (0..1)
  } },
  "raceControl":[{"utc":..,"lap":3,"category":"Flag","flag":"YELLOW","scope":"Sector","sector":4,"message":"...","num":null}],   // newest first, max 100
  "radio":[{"utc":..,"num":"44","tla":"HAM","url":"https://livetiming.formula1.com/static/<Path>"}],   // newest first, max 30
  "records":{"fastestLap":{"num":"1","sec":91.9,"text":"1:31.9","lap":9}|null,"topSpeed":{"num":..,"kph":..}|null},
  "pitStops":[{"num":"1","lap":14,"utc":..,"durationSec":22.4|null,"fromCompound":"SOFT","toCompound":"HARD"}]
}
```
### `{"type":"fast","utc":..,"cars":{"1":{"x":..,"y":..,"z":..,"speed":..,"rpm":..,"gear":..,"throttle":..,"brake":..,"drs":true,"progress":0.43}}}` — 4 Hz, only cars that changed.
### `{"type":"track","track":{"points":[[x,y],...],"length":5412.0,"bounds":[minx,miny,maxx,maxy],"startIdx":0,"drsZones":[[startFrac,endFrac]],"sectors":[frac1,frac2],"rotation":0}}` — sent when (re)built; also GET `/api/track`.
### `{"type":"history", ...}` — on connect and after each completed lap (throttled ≤0.5 Hz):
```jsonc
{ "type":"history","maxLap":12,
  "laps": { "1":[{"lap":1,"sec":95.1|null,"s":[28.1,31.0,36.0],"compound":"SOFT","tyreAge":1,"pos":1,"gapLeader":0.0|null,"interval":..|null,"pit":false,"track":"1"}] },
  "events":[{"utc":..,"lap":3,"kind":"flag|sc|vsc|pit|fastest|retire|penalty|radio|overtake","num":"44"|null,"text":"..."}]   // chronological, max 500
}
```
REST: `GET /api/state` `GET /api/history` `GET /api/track` `GET /api/status` `POST /api/control {speed,paused,seek}`. Static frontend served at `/`.
Server entry: `python -m backend --mode sim|file|live [--file path] [--speed 1] [--port 8000]`.

## 4. Frontend module contract (vanilla ES modules, NO build step, NO external CDN; charts hand-drawn on canvas/SVG)
`frontend/js/store.js` (provided) exports a singleton `store`:
`store.state` `store.fast` `store.track` `store.history` `store.hello` (latest messages; cars merged),
`store.selected` (Set of car nums; first = primary), `store.subscribe(fn)`→unsub (fn(kind) kind∈'state'|'fast'|'track'|'history'|'selection'|'conn'),
`store.select(num, {additive})`, `store.send(obj)`, `store.conn` ('connecting'|'open'|'closed'). Panels: `export function mount(el, store)`.

## 5. Clarifications added by the engine/server implementation (additive, no breaking changes)
* `brake` (in `tel` and `fast`) is normalised to `0` or `100` (same scale as throttle), whatever the raw 0/1 or 0/100 feed sent.
* `drsZones` entries may wrap over the start/finish line (`startFrac > endFrac`, e.g. `[0.96, 0.04]`). `sectors` = the two boundary fractions.
  The server re-sends `{"type":"track"}` whenever the track is (re)built or its DRS zones / sector fractions change.
* `track.length` is metres (raw position units are decimetres; the builder rescales when the loop is > 15000 raw units).
* `GET /api/track` returns the same envelope as the WS frame: `{"type":"track","track":{...}|null}`.
* `GET /api/status` -> `{mode, replay, clients, msgsPerSec, totalMsgs, lastMessageAgeSec, ended, error, uptimeSec, control, engine:{drivers,laps,events,pitStops,radio,raceControl,msgs,errors,ignored,trackBuilt,utc,topics}, trackVersion, utc}`.
* Extra WS frame `{"type":"control","speed":..,"paused":..}` is broadcast to all clients after any control change (clients may ignore it).
* A seek resets the engine (track geometry is kept) and the source re-sends everything up to the target; the server then pushes state/history/track to all clients immediately.
* Leader gap fields read `"+0.000"` / `0.0` when the feed sends an empty string; lapped cars keep their raw text (`"1L"`) with `null` seconds. Empty sector cells have `text: null, sec: null`.

## 6. Non-race sessions (practice / qualifying / sprint) — shapes recalled from the real feed; BE LENIENT (any field may be missing)
Raw additions (same topics as §2):
* `SessionInfo`: `Type` ∈ "Practice"|"Qualifying"|"Race"; `Name` ∈ "Practice 1/2/3"|"Qualifying"|"Sprint Qualifying"|"Sprint Shootout"|"Sprint"|"Race"; `Meeting.Circuit.Key`, `Number`. (Sprint: Type "Race", Name "Sprint". Sprint quali: Type "Qualifying".)
* `TimingData` top level may carry `SessionPart` (1|2|3, quali parts), `Withheld`, `CutOffTime`. `Lines[n]` in practice/quali carry **`TimeDiffToFastest`** ("+0.234", "" for P1) and **`TimeDiffToPositionAhead`** ("+0.056") instead of `GapToLeader`/`IntervalToPositionAhead`; `BestLapTime:{Value,Lap}`; `BestLapTimes` (quali: list/dict of 3 `{Value,Lap}` per part); `KnockedOut:bool`; `Cutoff:bool`; `Stats`; `Status:int`; `NumberOfLaps` counts timed laps only. `Position` is by best lap. Cars that have not set a time have no `BestLapTime`/`Position` may be stale.
* No `LapCount` in practice/quali. `ExtrapolatedClock.Remaining` is the session/part clock. `SessionStatus` cycles "Started"/"Aborted"(red flag)/"Finished"/"Finalised"/"Ends" per quali part. `SessionData.Series/StatusSeries` may appear (lists of `{Utc, Lap|TrackStatus|SessionStatus}`).
* Cars in the garage: `Position` OffTrack/`InPit:true`; they can sit for 10+ minutes with no new laps.
Output additions to `state` (existing fields keep working; for non-race sessions the engine MUST fill `gapLeader/gapLeaderSec/interval/intervalSec` from TimeDiffToFastest/TimeDiffToPositionAhead so old UI still works):
* `session.kind`: "practice"|"qualifying"|"sprint_qualifying"|"sprint"|"race"|"unknown" (derive from Type+Name, case-insensitive; "Shootout" counts as sprint_qualifying); `session.part`: 1|2|3|null; `session.partLabel`: "Q1"|"Q2"|"Q3"|null.
* `lap.total` is null when there is no LapCount; `lap.current` may be null/0 — UI must not show "0 / —" in non-race sessions.
* per car: `gapFastest` (text), `gapFastestSec`, `knockedOut` bool, `cutoff` bool, `bestLaps` [{text,sec}×3 or nulls], `activity` ∈ "track"|"pit"|"out"|"garage"|"retired" (garage = in pit with no movement/ no timing change for 60 s+ in non-race sessions), `timedLaps` (= NumberOfLaps).
* `history.laps` entries for practice/quali: include in/out laps (`sec:null` allowed) and a `kind`-agnostic shape; `pit` true for in/out laps. Overtake detection OFF for non-race sessions (order changes by lap times, not passes). "Fastest" events fire when a driver takes P1.
