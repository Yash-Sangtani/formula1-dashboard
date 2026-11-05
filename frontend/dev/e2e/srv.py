import sys, logging
sys.path.insert(0, '/home/user/formula1-dashboard')
import uvicorn
from backend.server import create_app
from backend.ingest.sources import SimSource
port, session, speed = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
ts = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
kw = {'session': session}
if session in ('fp', 'quali', 'sprint_quali'): kw['time_scale'] = ts
laps = 10 if session in ('race', 'sprint') else 20
src = SimSource(seed=3, speed=speed, laps=laps, **kw)
app = create_app(src, mode='sim')
uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
