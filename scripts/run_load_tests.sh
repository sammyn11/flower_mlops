#!/usr/bin/env bash
# Runs Locust against 1, 2 and 3 API containers x 50/100/200 users, then summarises.
set -e
mkdir -p results
for N in 1 2 3; do
  echo "=== $N API container(s) ==="
  docker compose up -d --scale api=$N --no-recreate
  docker compose restart nginx          # nginx re-resolves the api containers
  until curl -sf http://localhost:8080/status >/dev/null; do sleep 3; done
  sleep 10
  for U in 50 100 200; do
    locust -f locustfile.py --host http://localhost:8080 --headless \
      -u $U -r 10 -t 60s --csv results/c${N}_u${U} --only-summary
  done
done
python scripts/summarize_results.py
