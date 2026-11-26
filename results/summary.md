# Locust load-test summary

This is the final warmed local Docker benchmark from 2026-11-26. Locust 2.29.1 ran each case for 60 seconds through Nginx. The API was warmed before each run. Request timeout: 15 seconds. Requests/s includes timed-out attempts.

| Containers | Users | Requests | Failures | Failure rate | Average ms | Median ms | p95 ms | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 395 | 0 | 0% | 5290 | 6700 | 7600 | 6.67 |
| 1 | 100 | 383 | 158 | 41.25% | 11278 | 15000 | 15000 | 6.45 |
| 1 | 200 | 652 | 531 | 81.44% | 13867 | 15000 | 15000 | 11.05 |
| 2 | 50 | 551 | 0 | 0% | 3561 | 4300 | 5300 | 9.30 |
| 2 | 100 | 537 | 0 | 0% | 8096 | 8900 | 13000 | 9.04 |
| 2 | 200 | 664 | 515 | 77.56% | 13574 | 15000 | 15000 | 11.38 |
| 3 | 50 | 608 | 0 | 0% | 3145 | 3700 | 5100 | 10.25 |
| 3 | 100 | 533 | 1 | 0.19% | 8272 | 10000 | 13000 | 8.98 |
| 3 | 200 | 704 | 463 | 65.77% | 12645 | 15000 | 15000 | 11.88 |

At 50 users, more replicas increased throughput and reduced latency without failures. At 100 users, two replicas performed best, with zero timeouts. At 200 users, all configurations saturated the local CPU and exceeded the 15-second client timeout frequently; three replicas reduced the failure rate compared with one or two. These results are for local Docker, not cloud production. Full interpretation: `benchmark_summary.md`.
