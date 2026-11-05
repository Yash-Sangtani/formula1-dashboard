Playwright checks (NODE_PATH=$(npm root -g) node <file>). Serve frontend/ statically, e.g. python3 -m http.server 8150 --directory frontend
  fuzz.js  <base> [race,fp,quali] [iterations]   malformed frames -> no console errors / panel-error badges
  life.js  soak.js                               cars appearing / driver list change / kind switch; 3 h memory soak
  shots.js <outdir> <base> <kinds> <views>       screenshots from ?mock=race|fp|quali|sprint_quali|sprint (add &warm=100 for FP/Q)
  srv.py <port> <fp|quali|sprint_quali|sprint|race> <speed> [time_scale]   real backend sim server; real.js drives it
