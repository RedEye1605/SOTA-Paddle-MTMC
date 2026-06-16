# Benchmark Plan

## Scenarios

| # | Cameras | Topology | Goal |
|---|---|---|---|
| 1 | 2 | same area, different angle | cross-angle re-id |
| 2 | 2 | sequential path, valid transition | travel-time filter |
| 3 | 2 | impossible direct transition | camera_links hard-block |
| 4 | 4 | mixed angle + distance | scaling test |
| 5 | 2 | long session simulation | 24 h identity window |

## Metrics (all required)

```
per_camera_analytics_fps
total_analytics_fps
gpu_memory_used_mb
cpu_usage_percent
local_id_switches_per_tracklet
cross_camera_match_accuracy      (precision/recall on labeled set)
false_merge_rate                  (1 - precision of "same global_id" claim)
id_fragmentation_rate             (1 - recall of "same global_id" claim)
ambiguous_decision_rate
qdrant_query_latency_p50_p95_p99_ms
postgres_write_latency_p50_p95_p99_ms
redis_stream_backlog
```

## Scripts

- `scripts/benchmark_t4.py` — runs scenarios 1–5 against a recorded dataset
- `scripts/compare_with_service_baseline.py` — side-by-side comparison with `Service/`
  on a shared dataset
- `Docs/benchmark_results_t4.md` — table of results
- `Docs/recommendation_after_benchmark.md` — chosen configuration after benchmark

## Acceptance

Phase 10 is complete when:

1. 4 cameras run concurrently at ≥5 FPS each on T4
2. Cross-camera match accuracy ≥ 0.85 on scenario 1
3. False merge rate ≤ 0.05
4. ID fragmentation rate ≤ 0.20 (some fragmentation is acceptable; false merges are not)
5. Qdrant p99 query latency ≤ 100 ms
6. PostgreSQL p99 write latency ≤ 50 ms
