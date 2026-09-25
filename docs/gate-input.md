# Deployment gate input

`quantserve gate` compares two complete end-to-end measurement summaries. Each
YAML file must be a mapping with these fields:

```yaml
measurement_source: measured_e2e
measurement_status: complete
ttft_p95_ms: 500.0
tpot_p95_ms: 40.0
throughput_tokens_per_s: 100.0
peak_vram_gb: 5.0
quality_retention_pct: 99.0
```

The baseline and candidate must use the same model, workload, hardware, quality
evaluation, and measurement method for a meaningful comparison. The gate checks
the declared source and completeness fields; it cannot independently verify how
the measurements were collected.

`throughput_tokens_s` is also accepted for throughput, and `quality_pct` for
quality. If both names for one metric are present, their values must agree.
Quality is expressed from 0 to 100; quality drop thresholds are percentage
points. Latency and throughput thresholds are relative percentages. VRAM
thresholds are absolute GB. Defaults allow at most 10% higher P95 TTFT or TPOT,
10% lower throughput, no peak VRAM increase, and a 2 percentage point quality
drop. Adjust these limits with `quantserve gate --help`.

```bash
quantserve gate --baseline production.yaml --candidate candidate.yaml
```

Exit code 0 means all five measured criteria pass; 1 means a measured regression;
2 means missing, malformed, synthetic, or incomplete input. A recommendation
file with `predicted_metrics` is not a measured gate input. The serving benchmark
also does not currently collect quality or peak VRAM, so its output alone is
insufficient. Assemble complete summaries from measured serving, memory, and
quality evaluations before using this as a deployment check.
