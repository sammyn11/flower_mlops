# Locust flood test results

Run date: 2026-11-26. Locust 2.29.1; each run lasted 60 seconds with the indicated number of concurrent users. API replicas were warmed before each measurement. The Locust workload sent image predictions and status checks through Nginx. The configured client timeout is 15 seconds. Requests/s counts all attempts, including timed-out requests.

| API containers | Users | Requests | Failures | Failure rate | Average (ms) | Median (ms) | p95 (ms) | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 395 | 0 | 0% | 5,290 | 6,700 | 7,600 | 6.67 |
| 1 | 100 | 383 | 158 | 41.25% | 11,278 | 15,000 | 15,000 | 6.45 |
| 1 | 200 | 652 | 531 | 81.44% | 13,867 | 15,000 | 15,000 | 11.05 |
| 2 | 50 | 551 | 0 | 0% | 3,561 | 4,300 | 5,300 | 9.30 |
| 2 | 100 | 537 | 0 | 0% | 8,096 | 8,900 | 13,000 | 9.04 |
| 2 | 200 | 664 | 515 | 77.56% | 13,574 | 15,000 | 15,000 | 11.38 |
| 3 | 50 | 608 | 0 | 0% | 3,145 | 3,700 | 5,100 | 10.25 |
| 3 | 100 | 533 | 1 | 0.19% | 8,272 | 10,000 | 13,000 | 8.98 |
| 3 | 200 | 704 | 463 | 65.77% | 12,645 | 15,000 | 15,000 | 11.88 |

At 50 users, adding replicas increased measured request throughput from 6.67 requests/s with one container to 10.25 requests/s with three, while reducing average latency from 5.29 s to 3.15 s; all three configurations had no failures. At 100 users, one replica timed out on 41.25% of requests, while two replicas had no failures and three had one timeout. At 200 users, the host was saturated: failure rates were 81.44%, 77.56%, and 65.77% for one, two, and three replicas. Three containers provided the best result at this load, but the CPU-limited local Docker host could not reliably serve 200 concurrent users. These are local Docker measurements, not cloud production measurements.

The 15-second percentiles at higher loads reflect client timeouts; most errors were Locust `ReadTimeout` errors. Cold-start inference was much slower than warmed inference, so all reported runs were warmed before collection.
