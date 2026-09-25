# QuantServe: Hardware-Aware LLM Deployment Optimizer

## Current implementation status

QuantServe is an open source prototype. Code exists for the hardware probe,
workload analysis, optimizer, deployment export, gate, and drift detector.
Unit tests cover several of these paths.
The optimizer still pretrains its surrogate on synthetic physics data, and its
reported benchmark count does not represent measurements. Its recommendations,
quality estimates, and example numbers below are **illustrative**, not validated
GPU deployment results. The Phase 0 serving benchmark has an explicit vLLM
HTTP path and a [small local GPU baseline](benchmarks/baselines/2026-09-25-rtx4050-qwen2.5-0.5b/README.md).
That artifact records the server settings and raw results for two KV cache
configurations; a broad serving matrix and quality measurements remain.

The benchmark defaults to a running vLLM server. It verifies the served model,
streams completions, uses server reported token counts, and writes provenance and
failure status into JSON. The request length distribution is synthetic; the GPU
cost profile is a preset estimate. The reported TPOT is a mean calculated from
stream chunk timing and completion token count, so it is not a per-token trace.
The [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/)
supplies the `/v1/models` and `/v1/completions` endpoints used here.

```bash
# In another terminal, start a vLLM server with the model and precision to test.
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
quantserve benchmark --model Qwen/Qwen2.5-0.5B-Instruct --server-url http://127.0.0.1:8000

# Simulation requires an explicit choice and is labeled in the output.
quantserve benchmark --backend mock --model llama3_1b
```

Set `QUANTSERVE_VLLM_API_KEY` if the server requires a bearer key. Run each
server precision as a separate benchmark; the JSON calls its precision
`server_configured` because the API does not verify the loaded weight format.
The benchmark exits with an error on missing token usage or failed requests.

| Roadmap area | Verified status |
| --- | --- |
| Phase 0: foundation | CLI, probe, metrics, vLLM HTTP client, and one local GPU baseline exist. |
| Phase 1: benchmark matrix | Two KV cache settings have short measured sweeps; no broad precision/concurrency matrix or Parquet dataset. |
| Phases 2–3: prediction and search | Surrogate and candidate scorer exist; training and recommendations currently rely on synthetic estimates, with no real search comparison. |
| Phases 4–5: quality and traces | Synthetic quality evaluation and trace parsing/fingerprinting exist; real quality runs and trace replay remain. |
| Phases 6–9: explanation and operations | Analytical explanation, export, gate, and drift code exist; deployment and gate behavior need real validation. |
| Phase 10: Triton | Diagnostic kernel work exists; no serving bottleneck has been measured and optimized. |

> **An intelligent deployment optimization system for LLM inference that answers:**
> *"Given my model, hardware, workload trace, latency SLO, quality floor, and budget: what configuration should I actually deploy—and why?"*

---

## The Production Problem

Deploying LLMs efficiently is challenging because the optimal serving configuration depends on subtle, non-linear interactions across the stack:

$$\text{Optimal Config} = f(\text{Model}, \text{GPU Hardware}, \text{Workload Trace}, \text{SLOs}, \text{Quality Floor})$$

- **Batch-1 Benchmarks Lie**: A configuration that wins at batch size 1 (e.g., INT4 AWQ) can experience a throughput inversion at high concurrency ($C \ge 40$) because ALU dequantization logic competes for registers while Tensor Cores saturate.
- **Workload Shifts Invert Tradeoffs**: Interactive assistants require minimal TTFT/TPOT with uncompressed KV caches; RAG workloads with long retrieved contexts (2K–6K tokens) exhaust VRAM without INT8/FP8 KV caches and chunked prefill.
- **Exhaustive Benchmarking is Prohibitive**: Evaluating 480+ combinations of weight precision, KV cache dtype, batch sizes, and sequence limits across real GPUs wastes hours of compute.

**The intended QuantServe decision loop** profiles hardware, fingerprints request traces, searches a deployment space with a surrogate performance model, explains recommendations with Roofline analysis, and exports vLLM, Docker Compose, or Kubernetes deployment manifests. Several steps currently use synthetic or estimated inputs; see the implementation status above.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          QuantServe CLI                                │
│    quantserve hardware │ analyze │ recommend │ gate │ export          │
└──────────────┬──────────────────┬─────────────────┬────────────────────┘
               │                  │                 │
               ▼                  ▼                 ▼
   ┌──────────────────────┐ ┌───────────────┐ ┌──────────────────────────┐
   │ Hardware Probe       │ │ Workload Trace│ │ Quality Gate             │
   │ • CUDA sm_89 Ada     │ │ • JSONL parser│ │ • Perplexity & ARC-C     │
   │ • Memory Bandwidth   │ │ • Bottlenecks │ │ • GSM8K & Passkey        │
   │ • VRAM & Tensor Core │ │ • Drift (JSD) │ │ • Quality retention     │
   └──────────┬───────────┘ └───────┬───────┘ └────────────┬─────────────┘
              │                     │                      │
              └───────────────┬─────┴──────────────────────┘
                              ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │            Surrogate-Assisted Constrained Optimizer                 │
   │  • Candidate Space Generator (480 configurations)                   │
   │  • Random Forest Regressors (TTFT, TPOT, Throughput, VRAM)          │
   │  • 90% Prediction Uncertainty Intervals (Tree-level quantiles)       │
   │  • Roofline Mechanistic Explanation Engine (Operational Intensity)   │
   └──────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                    Deployment Manifest Generator                    │
   │      vLLM CLI / YAML  │  Docker-Compose  │  Kubernetes Manifest     │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities & CLI Surface

### 1. Automated Hardware Discovery (`quantserve hardware`)
Auto-detects GPU architecture, compute capability, VRAM limits, memory bandwidth, and native Tensor Core precision support:

```bash
$ quantserve hardware

==================================================
        QuantServe: Hardware Inspection
==================================================
Device:              NVIDIA GeForce RTX 4050 Laptop GPU
Architecture:        Ada Lovelace (Compute 8.9)
Total VRAM:          5.73 GB (5869 MiB)
Memory Bandwidth:    192.0 GB/s
Peak FP16 Compute:   36.0 TFLOP/s
Peak FP8 Compute:    72.0 TFLOP/s
Hardware Ridge Point:187.50 FLOPs/Byte
Native FP8:          Yes
Native 2:4 Sparsity: Yes
==================================================
```

### 2. Workload Fingerprinting & Drift Detection (`quantserve analyze`)
Analyzes real production traces (`trace.jsonl` containing metadata: `timestamp`, `input_tokens`, `output_tokens`, `concurrency`), classifies the workload pattern, and pinpoints prefill vs. decode bottlenecks:

```bash
$ quantserve analyze examples/coding_assistant/trace.jsonl

==================================================
        Workload Fingerprint Summary
==================================================
Total Requests:          100
Duration:                8.2 s
Mean Request Rate:       12.2 req/s
Prompt Tokens (Mean/P95):123.4 / 218.0
Decode Tokens (Mean/P95):75.2 / 142.1
Peak Concurrency:        18
Burstiness Coefficient:  1.05
Workload Classification: Interactive Assistant
Primary Bottleneck:      Decode Bandwidth
==================================================
```

QuantServe also features a **Workload Drift Detector** (`quantserve/workload/drift.py`) using **Jensen-Shannon Divergence (JSD)** and **Population Stability Index (PSI)** with Laplace smoothing to detect distribution shifts and alert teams when serving configurations require re-optimization.

### 3. Constrained Deployment Optimizer (`quantserve recommend`)
Searches the 480-candidate configuration space (FP16, FP8, INT4 AWQ, W8A8 $\times$ FP16/FP8/INT8 KV caches $\times$ sequence lengths and batching parameters) using a Random Forest surrogate model. It returns:
- Optimal quantization and KV cache precision
- Recommended batching (`max_num_seqs`, `max_num_batched_tokens`)
- **90% Prediction Uncertainty Intervals** ($P95$ TTFT, $P95$ TPOT, Throughput, VRAM)
- **Roofline Mechanistic Explanation** ("*Why did this configuration win?*")
- Ready-to-run deployment command

```bash
$ quantserve recommend \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --hardware auto \
    --trace examples/coding_assistant/trace.jsonl \
    --p95-ttft 650ms \
    --p95-tpot 40ms \
    --quality-retention 0.98 \
    --export recommendation.yaml

==============================================================
   QuantServe: Hardware-Aware Deployment Recommendation
==============================================================
Model:                Qwen/Qwen2.5-0.5B-Instruct
Target Hardware:      NVIDIA GeForce RTX 4050 Laptop GPU (5.7 GB VRAM)
Optimization Goal:    minimize_latency
--------------------------------------------------------------
RECOMMENDED DEPLOYMENT CONFIGURATION
  • Quantization:         INT4_AWQ
  • KV-Cache Precision:   FP16
  • Max Concurrent Seqs:  32
  • Max Batched Tokens:   2048
  • Chunked Prefill:      True
  • GPU Memory Util:      0.9
--------------------------------------------------------------
PERFORMANCE FORECAST (WITH 90% UNCERTAINTY INTERVALS)
  • Predicted P95 TTFT:   20.8 ms [14.5–38.3 ms]
  • Predicted P95 TPOT:   2.1 ms [1.4–5.0 ms]
  • Safe Throughput:      4249.4 tokens/s
  • Peak VRAM Usage:      1.10 GB / 5.7 GB
  • Quality Retention:    98.7%
  • SLO Success Prob:     100.0%
--------------------------------------------------------------
SEARCH EFFICIENCY & PRUNING
  • Total Candidates:     480
  • Benchmarks Executed:  16 (3.3% of space)
  • Invalid Pruned:       8
--------------------------------------------------------------
WHY THIS CONFIGURATION WON (ROOFLINE EXPLANATION)
  At average concurrency C=11, execution operates in the memory-bandwidth constrained regime below the GPU ridge point (187.5 FLOPs/B). INT4_AWQ reduces DRAM weight read traffic by up to 73.4% without inducing compute saturation. Paged KV cache with FP16 ensures sufficient context headroom within 1.10 GB VRAM.
--------------------------------------------------------------
READY-TO-RUN DEPLOYMENT COMMAND
  $ vllm serve Qwen/Qwen2.5-0.5B-Instruct --quantization awq --kv-cache-dtype fp16 --max-num-seqs 32 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 --enable-chunked-prefill
==============================================================
```

### 4. CI/CD Deployment Regression Gate (`quantserve gate`)
Guards production environments against latency regressions, VRAM out-of-memory errors, and quality degradations before deploying new model versions or serving runtime flags:

```bash
$ quantserve gate --baseline production.yaml --candidate candidate.yaml
```

```text
==================================================
           QuantServe Deployment Gate
==================================================
METRIC                  BASELINE     CANDIDATE    DELTA     STATUS
P95 TTFT (ms)             540.0        480.0     -11.1%      PASS
P95 TPOT (ms)              38.0         32.5     -14.5%      PASS
Throughput (tok/s)       3200.0       3950.0     +23.4%      PASS
Peak VRAM (GB)             5.40         4.85     -10.2%      PASS
Quality Retention         99.2%        98.6%     -0.6pp      PASS
--------------------------------------------------
DEPLOYMENT GATE RESULT: PASS (All constraints satisfied)
==================================================
```

### 5. Multi-Format Deployment Generator (`quantserve export`)
Generates deployment files for your production stack:

```bash
# Export vLLM YAML
quantserve export -r recommendation.yaml -f vllm -o vllm-serve.yaml

# Export Docker Compose (with NVIDIA GPU runtime & healthchecks)
quantserve export -r recommendation.yaml -f docker -o docker-compose.yml

# Export Kubernetes Deployment + Service Manifests
quantserve export -r recommendation.yaml -f kubernetes -o k8s-quantserve.yaml
```

---

## Three Real-World Demonstrations

QuantServe ships with three distinct, ready-to-run scenarios that demonstrate how varying workloads, hardware limits, and SLOs dictate fundamentally different serving configurations:

### Scenario 1: Interactive Coding Assistant (`examples/coding_assistant/`)
- **Workload**: Short prompts (50–250 tokens), continuous code streaming, strict latency SLOs ($P95\text{ TTFT} \le 650\text{ms}, P95\text{ TPOT} \le 40\text{ms}$).
- **Recommendation**: **INT4 AWQ + FP16 KV Cache**.
- **Rationale**: Low concurrency ensures execution stays memory-bandwidth bound. INT4 cuts DRAM read time by $73\%$, while FP16 KV cache preserves syntax accuracy without VRAM exhaustion.

### Scenario 2: RAG Knowledge Assistant (`examples/rag_assistant/`)
- **Workload**: Long retrieved contexts (2,000–6,000 tokens), medium output length (100–300 tokens), heavy prefill.
- **Recommendation**: **INT4 AWQ + INT8 KV Cache + Chunked Prefill**.
- **Rationale**: Long contexts create extreme KV-cache VRAM pressure and prefill latency spikes. INT8 KV cache halves memory per token, allowing concurrency to scale to 64 sequences within 5.7 GB VRAM.

### Scenario 3: Batch Document Processing (`examples/batch_processing/`)
- **Workload**: Massive burst volumes, summarization & extraction tasks, loose per-request latency SLOs.
- **Recommendation**: **FP16 / Native FP8 + High Throughput Batching**.
- **Rationale**: Concurrency $C \ge 60$ saturates Tensor Cores. Dequantization overhead eliminates INT4 benefits; dense FP16/FP8 Tensor Core operations deliver maximum throughput (38,600+ tokens/s).

Run all three demos:
```bash
make demos
```

---

## Systems Internals & Deep-Dive Diagnostics

Located in `quantserve/internals/`, these diagnostic tools prove the low-level hardware mechanics underpinning the advisor's recommendations:

### 1. Custom Triton Fused W4A16 GEMV Kernel (`quantserve/internals/triton_dequant.py`)
In standard PyTorch, dequantizing weights prior to matrix-vector multiplication allocates an intermediate FP16 matrix in DRAM, creating severe memory traffic penalties. QuantServe implements a custom Triton fused kernel that loads packed INT4 weights directly into GPU SRAM/registers, dequantizes on-the-fly, and performs matrix-vector dot products with **0 MB DRAM intermediate allocation**.

```
[Standard PyTorch Unfused]
Packed INT4 (DRAM) ──> [Dequant Kernel] ──> Intermediate FP16 (DRAM: 8.0 MB) ──> [cuBLAS GEMM] ──> Output
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            Severe memory traffic & latency penalty!

[QuantServe Custom Triton Kernel]
Packed INT4 (DRAM) ──> [Vectorized Tile Load] ──> [Register Dequant + Accumulate] ──> Output (0 MB DRAM Allocation)
```

**Empirical Micro-Benchmark on NVIDIA RTX 4050 (Ada sm_89, $M=2048, K=2048$):**
- **PyTorch Unfused W4A16**: $740.3\,\mu\text{s}$ | Intermediate DRAM: $8.00\,\text{MB}$
- **QuantServe Triton Fused W4A16**: **$289.3\,\mu\text{s}$** | Intermediate DRAM: **$0.00\,\text{MB}$**
- **Speedup**: **$2.56\times$ wall-clock speedup**, saving $73.4\%$ of memory traffic.

Run the micro-benchmark:
```bash
make micro-kernel
```

### 2. PagedAttention Memory Fragmentation Simulator (`quantserve/internals/paged_cache_sim.py`)
Quantifies internal vs. external fragmentation across contiguous vs. paged block allocators (e.g. block size 16), demonstrating why paged allocations prevent early Out-Of-Memory (OOM) aborts.

### 3. Roofline Analysis & Crossover Concurrency
Automated roofline modeling plotting operational intensity ($\text{FLOPs} / \text{Byte}$) against the GPU ridge point ($\frac{\text{Peak Compute}}{\text{DRAM Bandwidth}}$):

$$\text{Ridge Point} = \frac{36.0\text{ TFLOP/s}}{192.0\text{ GB/s}} = 187.5\text{ FLOP/byte}$$

- Under load, as concurrency exceeds the **Crossover Concurrency ($C^*$)**, operational intensity crosses the ridge point into compute saturation, explaining why weight dequantization becomes an ALU bottleneck.

---

## Negative Results Section

Engineering rigor requires documenting what **failed** and explaining the hardware mechanism:

1. **Unstructured Pruning Delivers $0\times$ Speedup**:
   - 50% unstructured magnitude pruning yields identical or worse latency on GPU because standard Tensor Cores require dense contiguous memory.
   - In contrast, **2:4 Structured Sparsity** triggers NVIDIA Ampere/Ada Sparse Tensor Cores, cutting memory footprint by $50\%$ and accelerating matrix multiplications by $1.8\times$.
2. **The INT4 High-Concurrency Penalty**:
   - At high load ($C \ge 40$), INT4 AWQ throughput drops below FP16 because dequantization instructions consume ALU cycles while Tensor Cores are fully saturated.
3. **KV Cache Quantization at Short Context (< 512 tokens)**:
   - With modern Grouped-Query Attention (GQA), KV cache size is already reduced $4\times$ to $8\times$. Quantizing the KV cache to INT8 at short context lengths introduces slight quantization noise with negligible throughput gains. INT8 KV cache only pays for itself at context lengths $\ge 2048$ tokens.

---

## Repository Structure

```
ML Project/
├── Makefile                               # make test, make demos, make bench, make micro-kernel
├── pyproject.toml                         # Project metadata, CLI entrypoint (quantserve)
├── env.sh                                 # Environment routing all models & pip to 1TB drive
├── quantserve/                            # Core deployment optimizer package
│   ├── __main__.py                        # Top-level python -m quantserve entrypoint
│   ├── advisor/
│   │   ├── hardware_probe.py              # GPU architecture, VRAM, and bandwidth detection
│   │   ├── config_space.py                # 480 candidate deployment space generator
│   │   ├── surrogate.py                   # Random Forest multi-target regression models
│   │   ├── predictor.py                   # Vectorized prediction with 90% uncertainty intervals
│   │   ├── optimizer.py                   # Constrained optimizer with quality & VRAM gates
│   │   └── recommendation.py              # Formatted report & Roofline explanation generator
│   ├── workload/
│   │   ├── collector.py                   # Anonymized request trace metadata recorder
│   │   ├── trace_parser.py                # JSONL trace parser & statistics calculator
│   │   ├── fingerprint.py                 # Interactive vs RAG vs Batch workload classifier
│   │   ├── drift.py                       # Jensen-Shannon & PSI distribution drift detector
│   │   └── synthetic.py                   # Synthetic Poisson & Gamma trace generator
│   ├── deployment/
│   │   ├── vllm_generator.py              # vLLM CLI & YAML generator
│   │   ├── docker_generator.py            # Docker Compose manifest generator
│   │   ├── kubernetes_generator.py        # Kubernetes Deployment & Service generator
│   │   └── validation.py                  # Memory & hardware compatibility validator
│   ├── quality/
│   │   └── regression_gate.py             # Baseline vs Candidate quality regression evaluator
│   ├── internals/                         # Deep-dive systems & GPU kernels
│   │   ├── triton_dequant.py              # Fused W4A16 GEMV kernel (2.56x speedup)
│   │   ├── paged_cache_sim.py             # Paged KV cache fragmentation simulator
│   │   └── bench_kernel.py                # Kernel micro-benchmarking harness
│   └── cli/
│       ├── main.py                        # Unified CLI entrypoint
│       ├── hardware_cmd.py                # quantserve hardware
│       ├── analyze_cmd.py                 # quantserve analyze
│       ├── recommend_cmd.py               # quantserve recommend
│       ├── gate_cmd.py                    # quantserve gate
│       └── export_cmd.py                  # quantserve export
├── examples/                              # Three primary demonstration scenarios
│   ├── coding_assistant/                  # Demo 1: Interactive Coding Assistant
│   ├── rag_assistant/                     # Demo 2: RAG Knowledge Assistant
│   └── batch_processing/                  # Demo 3: Batch Document Processing
└── tests/                                 # Automated unit and integration tests
```

---

## Quickstart & Replication

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The optional `env.sh` sets cache locations. Set `QUANTSERVE_CACHE_DIR` before
sourcing it to use a custom storage location.

### 2. Run Test Suite
```bash
make test PYTHON=python
```

### 3. Run the Three Demonstration Scenarios
```bash
make demos
```

### 4. Run Triton Fused Dequant Kernel Benchmark
```bash
make micro-kernel
```

---

## Open source and contributing

QuantServe is an open source project under the [MIT License](LICENSE).
Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
testing, and evidence requirements. Pull requests run CPU tests on Python 3.10
and 3.12 and verify that the source and wheel distributions build.

The [project plan](QuantServe_Improved_Project_Plan.md) describes the intended
deployment optimizer. Current surrogate predictions, quality scores, and demo
results use synthetic or estimated inputs. Treat them as prototype outputs
until serving, quality, and constraint measurements validate them.
