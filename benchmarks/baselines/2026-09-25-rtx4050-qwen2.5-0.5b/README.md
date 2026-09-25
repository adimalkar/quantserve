# RTX 4050 local serving baseline (2026-09-25)

This is an **exploratory measured serving baseline**, collected with QuantServe
client code at commit `5e673ab`. The two raw sweep files are
[`fp16-kvfp16.json`](fp16-kvfp16.json) and [`fp16-kvfp8.json`](fp16-kvfp8.json).
Each contains three open loop arrival-rate points, server-reported token counts,
and `measurement_source: measured_e2e` with zero failed requests.

## Run context

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, compute capability 8.9 |
| Driver | 610.57.04 |
| vLLM / PyTorch | 0.30.0 / 2.13.0+cu132 |
| Model and tokenizer | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a7755` |
| Weight dtype | FP16 in both runs |
| KV cache | FP16 (`auto`) versus FP8 E4M3 (`fp8`), without calibrated scaling factors |
| Workload | [`configs/workload_baseline_small.yaml`](../../../configs/workload_baseline_small.yaml), SHA-256 `8be64b9882af762bca4136fc060a637ca3da9671c249384f1e74033375c488e8` |
| Timing | 8 second configured window per point; two warmup requests; seed 42 |
| Server limits | max model length 1024, max sequences 8, max batched tokens 1024, GPU memory utilization 0.65, eager execution |

The client ran from a separate Python 3.10 environment. The server ran in an
isolated Python 3.12 environment. To repeat the runs, install vLLM following
its [GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
start one server configuration at a time, and run the benchmark command after
the server reports healthy:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a7755 \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a7755 \
  --host 127.0.0.1 --port 8001 --dtype float16 \
  --max-model-len 1024 --gpu-memory-utilization 0.65 \
  --max-num-seqs 8 --max-num-batched-tokens 1024 \
  --enforce-eager --generation-config vllm

quantserve benchmark --config configs/workload_baseline_small.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm \
  --server-url http://127.0.0.1:8001 --output-dir outputs/fp16-kvfp16
```

For the second run, restart vLLM with `--kv-cache-dtype fp8` added and change
the output directory to `outputs/fp16-kvfp8`. Keep the two raw files in separate
directories because the benchmark's default filename is the same.

## Observed metrics

| Offered rate (req/s) | Requests | FP16 KV P95 TTFT / TPOT (ms) | FP8 KV P95 TTFT / TPOT (ms) | Total throughput (tok/s), both |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 63.75 / 19.14 | 33.28 / 10.42 | 40.75 |
| 2 | 17 | 49.24 / 19.73 | 42.53 / 13.23 | 150.88 |
| 4 | 39 | 50.43 / 17.83 | 69.82 / 18.04 | 324.38 |

Both runs processed identical request counts and identical prompt/output token
totals at each point. All requests met the configured P99 latency thresholds.

## Interpretation limits

These are short single runs with synthetic prompt lengths, not production trace
replay. Offered load stayed below observed capacity, so equal throughput mostly
reflects equal input traffic; it does not establish maximum serving throughput.
The FP8 run also selected a different vLLM attention backend, so the latency
differences cannot be attributed only to KV cache dtype. QuantServe derives
TPOT from streamed chunks and server token counts, not per-token timestamps.
The cost field in the raw JSON uses a hardware preset rather than a measured
machine price. Peak VRAM and model quality were not collected. In particular,
the uncalibrated FP8 KV run is **not** a validated deployment recommendation.
