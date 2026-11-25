"""Turn Locust CSVs into a markdown table for the README."""
import csv, glob, re
rows = []
for f in sorted(glob.glob("results/c*_u*_stats.csv")):
    n, u = re.search(r"c(\d+)_u(\d+)", f).groups()
    for r in csv.DictReader(open(f)):
        if r["Name"] == "Aggregated":
            rows.append((int(n), int(u), r))
rows.sort(key=lambda x: (x[1], x[0]))
lines = ["| Containers | Users | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Req/s |",
         "|---|---|---|---|---|---|---|---|"]
for n, u, r in rows:
    lines.append(f"| {n} | {u} | {r['Request Count']} | {r['Failure Count']} | "
                 f"{float(r['Average Response Time']):.0f} | {r['Median Response Time']} | "
                 f"{r['95%']} | {float(r['Requests/s']):.1f} |")
open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
