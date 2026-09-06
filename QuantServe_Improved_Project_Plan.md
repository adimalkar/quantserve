# QuantServe: Hardware-Aware LLM Deployment Optimizer

## Project Reframing

### Current Problem

The existing project is technically strong, but its main weakness is that it behaves primarily like an advanced benchmarking and systems research framework.

It can measure:

- Quantization performance
- TTFT / TPOT
- SLO goodput
- KV-cache behavior
- GPU utilization
- Roofline characteristics
- Concurrency crossovers
- Triton kernel performance
- Cost-per-token tradeoffs

However, the project currently ends mostly in:

- Benchmark tables
- Research findings
- Performance plots
- Profiling reports
- Compression comparisons

That makes it useful as a systems experiment, but much less useful as a real product or practical engineering tool.

The upgraded project should answer a concrete production question:

> **Given my model, hardware, traffic pattern, latency requirement, quality requirement, and budget, what is the best configuration I should actually deploy?**

---

# New Project Vision

## QuantServe: Hardware-Aware LLM Deployment Optimizer

QuantServe becomes an intelligent deployment optimization system for LLM inference.

Instead of asking engineers to manually benchmark dozens or hundreds of configurations, QuantServe automatically determines the best deployment configuration for their workload.

The system should take:

- Hugging Face model
- Available GPU hardware
- GPU memory
- Traffic traces
- Prompt-length distribution
- Output-length distribution
- Concurrency pattern
- TTFT requirement
- TPOT requirement
- Quality tolerance
- Context-length requirement
- Cost or hardware limits

And return:

- Recommended quantization method
- Recommended KV-cache precision
- Recommended batching configuration
- Recommended concurrency settings
- Maximum safe request rate
- Estimated P50/P95/P99 latency
- Estimated throughput
- Estimated VRAM usage
- Estimated quality regression
- SLO-compliance probability
- Recommended deployment command/configuration

---

# Example User Workflow

A user could run:

```bash
quantserve recommend \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --hardware auto \
  --trace traces/coding-assistant.jsonl \
  --p95-ttft 750ms \
  --p95-tpot 50ms \
  --quality-retention 0.98
```

QuantServe could return:

```text
Hardware detected
NVIDIA RTX 4050 Laptop GPU
VRAM: 6 GB

Candidate space
192 configurations

Benchmarks executed
24 / 192

Performance model
R² TTFT: 0.94
R² TPOT: 0.91

Recommended configuration
─────────────────────────
Quantization: AWQ W4A16
KV Cache: FP8
Serving Engine: vLLM
max_num_seqs: 16
max_num_batched_tokens: 2048

Predicted P95 TTFT: 612 ms
Predicted P95 TPOT: 39 ms
Safe throughput: 4.8 req/s
Peak VRAM: 5.41 GB

Quality retention: 98.7%
SLO probability: 96.8%

Alternatives evaluated: 24
Configurations eliminated: 168
```

The user could then export the result:

```bash
quantserve export recommendation.yaml
```

Or generate a deployment command:

```bash
quantserve deploy recommendation.yaml
```

---

# Core Real-World Problem

Deploying LLMs efficiently is difficult because the best configuration depends on several interacting factors:

- Model architecture
- Quantization method
- GPU architecture
- GPU memory
- Memory bandwidth
- Tensor Core performance
- Prompt length
- Generation length
- Request rate
- Concurrency
- Batch size
- KV-cache precision
- Latency requirements
- Quality requirements

There is no single universally optimal configuration.

A setting that is excellent for a batch-processing workload may perform poorly for an interactive assistant.

A quantization format that helps at low concurrency may become slower at high concurrency.

A lower precision KV cache may improve capacity but reduce long-context retrieval quality.

QuantServe should automatically navigate these tradeoffs.

---

# Core Research Question

The upgraded project's central research question becomes:

> **Can we learn the relationship between model architecture, compression, hardware, workload, and serving configuration well enough to recommend near-optimal LLM deployments while running only a fraction of the exhaustive benchmarks?**

This is stronger than simply asking whether one quantization technique is faster than another.

It combines:

- Machine learning
- GPU systems
- Inference engineering
- Optimization
- Performance modeling
- Production deployment
- SLO management

---

# Major Upgrade 1: Performance Prediction Model

The project should include an actual learned ML component.

Instead of exhaustively benchmarking every possible deployment configuration, QuantServe collects benchmark results and trains models that predict deployment performance.

The model should approximate:

\[
(model,\ hardware,\ quantization,\ workload,\ serving\ configuration)
\rightarrow
(TTFT,\ TPOT,\ throughput,\ VRAM,\ power)
\]

## Input Features

### Model Features

- Parameter count
- Number of transformer layers
- Hidden dimension
- Number of attention heads
- Number of KV heads
- GQA ratio
- Vocabulary size
- Maximum context length
- Model family
- Dense/MoE architecture

### Compression Features

- Weight precision
- Activation precision
- KV-cache precision
- Quantization algorithm
- Group size
- Sparsity ratio
- Sparse/dense execution

### Hardware Features

- GPU architecture
- CUDA compute capability
- VRAM
- Memory bandwidth
- SM count
- Tensor Core throughput
- FP16 performance
- FP8 performance
- INT8 performance

### Workload Features

- Average input sequence length
- P95 input sequence length
- Average output sequence length
- P95 output sequence length
- Request rate
- Concurrency
- Burst coefficient
- Prompt/output ratio

### Runtime Features

- max_num_seqs
- max_num_batched_tokens
- chunked prefill
- GPU memory utilization
- scheduler configuration
- serving backend
- attention backend

## Prediction Targets

Separate models may be trained for:

- TTFT
- TPOT
- Throughput
- Peak VRAM
- Power consumption
- SLO compliance probability

Suitable initial models include:

- XGBoost
- LightGBM
- Random Forest
- Gradient Boosting

There is no need to force a neural network into the project.

A tree-based regression model is likely a better fit for the structured configuration space.

---

# Major Upgrade 2: Prediction Uncertainty

QuantServe should not return a single prediction without confidence information.

The system should report uncertainty, for example:

```text
Predicted P95 TTFT: 640 ms
90% prediction interval: 588–703 ms
```

Possible approaches:

- Quantile regression
- Conformal prediction
- Bootstrap intervals
- Ensembles

This makes the deployment recommendation much more trustworthy.

---

# Major Upgrade 3: Intelligent Configuration Search

The configuration space becomes large very quickly.

For example:

```text
4 quantization schemes
3 KV-cache precisions
8 concurrency settings
8 batching configurations
4 context lengths
3 serving configurations
```

This already produces:

\[
4 \times 3 \times 8 \times 8 \times 4 \times 3 = 9,216
\]

possible configurations.

Benchmarking all of them is expensive and unnecessary.

QuantServe should use intelligent search.

Possible techniques:

- Bayesian optimization
- Optuna
- Tree-structured Parzen Estimators
- Active learning
- Successive halving

The system should benchmark a small number of configurations, train/update its surrogate performance model, and choose the next most informative or promising configuration.

## Important Evaluation Metric

Measure:

> How many real GPU benchmark runs are required before QuantServe finds a configuration close to the exhaustive-search optimum?

Useful experimental metrics include:

- Percentage of configuration space evaluated
- Distance from exhaustive optimum
- Total benchmarking time
- Number of GPU-hours saved
- Recommendation accuracy
- Regret

A strong experimental result would show that QuantServe reaches near-optimal deployment configurations while testing only a fraction of all possible configurations.

---

# Major Upgrade 4: Production Workload Traces

Synthetic Poisson and Gamma traffic should remain for reproducibility, but they should not be the primary workflow.

QuantServe should support capturing anonymized production traffic characteristics.

The trace collector should record metadata such as:

```text
timestamp
input_tokens
output_tokens
TTFT
TPOT
request_duration
active_concurrency
```

The actual user prompt content does not need to be stored.

Example:

```text
APPLICATION
    │
    ▼
QuantServe Trace Collector
    │
    ├── timestamp
    ├── input token count
    ├── output token count
    ├── TTFT
    ├── TPOT
    └── concurrency
    │
    ▼
trace.jsonl
```

The user could then run:

```bash
quantserve analyze trace.jsonl
```

Example output:

```text
Workload Fingerprint

Requests/hour:        18,420
Average prompt:          742 tokens
P95 prompt:            3,871 tokens
Average generation:      186 tokens
Burst coefficient:       2.7
Peak concurrency:         21

Dominant workload:
Long-prefill interactive RAG

Likely bottleneck:
Prefill compute

Recommended search focus:
FP8 / AWQ + chunked prefill
```

This makes QuantServe useful against real application traffic.

---

# Major Upgrade 5: Workload Classification

QuantServe should characterize the workload before optimization.

Example workload classes:

### Interactive Assistant

Characteristics:

- Low to moderate prompt size
- Short TTFT requirements
- Continuous token generation
- Moderate concurrency

Priority:

- TTFT
- TPOT
- responsiveness

### RAG / Knowledge Assistant

Characteristics:

- Large retrieved contexts
- Expensive prefill
- High KV-cache usage

Priority:

- Prefill latency
- memory capacity
- context handling

### Batch Processing

Characteristics:

- Large numbers of queued requests
- Loose per-request latency requirements

Priority:

- Throughput
- GPU utilization
- cost/token

The optimizer can use different objective functions depending on workload class.

---

# Major Upgrade 6: Quality Regression Gate

Performance alone should never determine the recommendation.

Every candidate configuration must satisfy:

\[
Performance\ SLO
\land
Memory\ Constraint
\land
Quality\ Constraint
\]

The fastest configuration is not useful if model quality degrades too much.

Example:

```text
Candidate                  Quality   TTFT     Cost    Result

FP16                       100.0%    910 ms   $$$     FAIL latency
FP8                         99.8%    670 ms   $$      PASS
AWQ INT4                    98.4%    520 ms   $       PASS ← BEST
GPTQ INT4                   96.2%    495 ms   $       FAIL quality
INT4 + FP8 KV               97.9%    480 ms   $       FAIL quality
```

The optimizer should choose the best configuration among those that remain valid.

## Evaluation Tasks

The evaluation framework can support:

- Perplexity
- ARC-Challenge
- GSM8K
- HumanEval / coding tasks
- RAG retrieval evaluation
- Needle-in-a-haystack tests
- Long-context passkey retrieval
- Application-specific custom evaluation sets

The user should also be able to provide a custom evaluation function.

---

# Major Upgrade 7: Deployment Generator

Once QuantServe finds the best configuration, it should produce something deployable.

Possible outputs:

- vLLM CLI command
- vLLM YAML config
- Docker Compose config
- Docker command
- Kubernetes deployment manifest
- Environment variables
- Recommended resource limits

Example:

```bash
vllm serve Qwen/Qwen2.5-Coder-3B-Instruct \
  --quantization awq \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90
```

This closes the gap between benchmarking and deployment.

---

# Major Upgrade 8: CI/CD Deployment Regression Gate

QuantServe should support validating proposed inference configuration changes before deployment.

Example:

```bash
quantserve gate \
  --baseline production.yaml \
  --candidate candidate.yaml
```

Example CI output:

```text
QuantServe Deployment Gate

Performance
P95 TTFT       614 ms → 527 ms     ✓ -14.2%
P95 TPOT      43.1 ms → 39.7 ms    ✓ -7.9%
Throughput     4.7 rps → 5.3 rps   ✓ +12.8%

Memory
Peak VRAM      5.61 GB → 5.24 GB   ✓

Quality
HumanEval      63.4% → 63.1%       ✓ -0.3pp
Passkey        98.2% → 97.9%       ✓

DEPLOYMENT GATE: PASS
```

The system could integrate with GitHub Actions and automatically comment on pull requests.

This turns QuantServe into practical inference infrastructure.

---

# Major Upgrade 9: Workload Drift Detection

Production workloads change over time.

For example:

```text
Original average prompt: 500 tokens
Current average prompt: 3,800 tokens
```

The original deployment configuration may no longer be optimal.

QuantServe should compare the current traffic distribution to the workload used during optimization.

Possible metrics:

- Jensen-Shannon divergence
- Wasserstein distance
- Population Stability Index
- Changes in sequence-length percentiles
- Changes in request-rate distribution
- Changes in concurrency

Example:

```text
WORKLOAD DRIFT DETECTED

Optimization workload
P95 ISL: 1,450 tokens

Current workload
P95 ISL: 5,720 tokens

Jensen-Shannon divergence: 0.41

The current deployment recommendation may be stale.

Re-optimization recommended.
```

This moves the project toward inference operations rather than one-time benchmarking.

---

# Major Upgrade 10: Hardware-Aware Optimization

Hardware should be treated as a first-class input.

QuantServe should automatically detect:

- GPU name
- GPU architecture
- Compute capability
- VRAM
- Memory bandwidth
- Supported precisions
- Tensor Core capabilities
- CUDA version
- Driver version

The optimizer should understand that the same model/configuration behaves differently across hardware.

Long-term hardware support could include:

- RTX 4050
- RTX 4090
- A10
- A100
- L4
- L40S
- H100
- H200

A particularly useful niche is:

> **Open-source LLM deployment optimization for constrained and commodity GPUs where exhaustive search is too expensive.**

This is especially valuable because many optimization systems focus on high-end datacenter hardware.

---

# Major Upgrade 11: Roofline Analysis as Explanation Engine

Roofline analysis should remain a major part of the system, but it should explain recommendations rather than serve as the final product.

Example:

```text
Why did QuantServe recommend AWQ?

At requested concurrency C=6:

Measured operational intensity:
19.7 FLOP/byte

Hardware regime:
Memory-bandwidth constrained

INT4 reduces weight traffic enough to offset
dequantization overhead.

Predicted crossover:
C* ≈ 23

Above C≈23, FP8 becomes preferable.
```

The system therefore provides both:

1. A recommendation
2. A mechanistic explanation

This is significantly more valuable than simply reporting that one configuration is faster.

---

# Major Upgrade 12: Crossover Concurrency

The existing crossover-concurrency idea remains one of the strongest research components.

Define:

\[
C^* = \text{concurrency where the preferred execution strategy changes}
\]

Example:

```text
Concurrency 1–22:
AWQ INT4 wins

Concurrency 23–48:
FP8 wins

Concurrency 49+:
Dense compute becomes competitive
```

The optimizer can incorporate this information when choosing deployment configurations.

---

# Major Upgrade 13: Cost-Aware Optimization

The optimizer should support cost constraints.

Possible objectives:

```text
Minimize:
GPU cost / valid token

Subject to:
P95 TTFT < threshold
P95 TPOT < threshold
quality retention > threshold
VRAM < hardware limit
```

Example:

```text
Objective:
Minimize estimated monthly serving cost

Constraints:
P95 TTFT < 750 ms
P95 TPOT < 50 ms
Quality retention >= 98%
Peak VRAM <= 6 GB
```

This makes the project directly relevant to infrastructure decisions.

---

# Major Upgrade 14: Pareto Frontier

Some deployments have multiple valid tradeoffs.

Instead of returning only one configuration, QuantServe should expose the Pareto frontier.

Example:

```text
Configuration A
Best latency

Configuration B
Best cost

Configuration C
Best quality

Configuration D
Best memory efficiency
```

The system should still choose one default recommendation based on the user's objective.

---

# Three Primary Real-World Demonstrations

The final repository should demonstrate QuantServe on real application scenarios rather than arbitrary benchmarks.

## Demo 1: Interactive Coding Assistant

Example model:

```text
Qwen2.5-Coder
```

Primary objectives:

- Low TTFT
- Low TPOT
- Interactive responsiveness

Important variables:

- Quantization
- batching
- concurrency
- KV-cache precision

---

## Demo 2: RAG Knowledge Assistant

Example workload:

- Large retrieved contexts
- Medium output length
- High prefill cost

Primary objectives:

- Prefill latency
- VRAM capacity
- long-context quality

Important variables:

- chunked prefill
- KV-cache precision
- context length
- batching

---

## Demo 3: Batch Document Processing

Example workload:

- Summarization
- extraction
- classification
- document generation

Primary objectives:

- Maximum throughput
- minimum cost/token

Latency is less important.

This scenario should produce a very different optimal configuration than the interactive assistant.

That difference demonstrates the project's core thesis:

> **There is no universally best LLM inference configuration. The optimal configuration depends on workload, hardware, quality requirements, and SLOs.**

---

# Recommended Repository Architecture

```text
quantserve/
│
├── advisor/
│   ├── hardware_probe.py
│   ├── config_space.py
│   ├── optimizer.py
│   ├── surrogate.py
│   ├── predictor.py
│   └── recommendation.py
│
├── workload/
│   ├── collector.py
│   ├── trace_parser.py
│   ├── fingerprint.py
│   ├── synthetic.py
│   ├── replay.py
│   └── drift.py
│
├── benchmark/
│   ├── runner.py
│   ├── metrics.py
│   ├── profiler.py
│   └── harness.py
│
├── compression/
│   ├── awq.py
│   ├── gptq.py
│   ├── fp8.py
│   ├── smoothquant.py
│   └── kv_cache.py
│
├── quality/
│   ├── evaluator.py
│   ├── task_suite.py
│   ├── regression_gate.py
│   └── custom_eval.py
│
├── analysis/
│   ├── roofline.py
│   ├── crossover.py
│   ├── pareto.py
│   └── cost.py
│
├── deployment/
│   ├── vllm_generator.py
│   ├── docker_generator.py
│   ├── kubernetes_generator.py
│   └── validation.py
│
├── internals/
│   ├── paged_cache_sim.py
│   ├── triton_dequant.py
│   └── bench_kernel.py
│
├── integrations/
│   └── github_actions.py
│
├── cli/
│   ├── recommend.py
│   ├── analyze.py
│   ├── benchmark.py
│   ├── gate.py
│   └── export.py
│
├── dashboard/
│
├── configs/
│
├── tests/
│
├── examples/
│   ├── coding_assistant/
│   ├── rag_assistant/
│   └── batch_processing/
│
├── pyproject.toml
├── Makefile
└── README.md
```

---

# Components to Keep From the Existing Project

The existing project already contains strong engineering work.

These components should remain:

- Open-loop load generator
- Poisson workloads
- Gamma burst workloads
- Trace replay
- TTFT metrics
- TPOT metrics
- P50/P95/P99 latency
- SLO goodput
- AWQ
- GPTQ
- FP8
- SmoothQuant
- KV-cache quantization
- Roofline analysis
- Operational intensity
- Crossover concurrency
- Cost-per-token analysis
- PyTorch profiling
- Nsight profiling
- Triton kernel experiments
- Paged KV-cache analysis
- Long-context quality testing
- Statistical confidence intervals

The difference is that these components become the underlying measurement and analysis engine for the deployment optimizer.

---

# Components to Remove or Demote

## Remove the Vision Transformer Track From the Main Project

The ViT compression track makes the repository broader but less coherent.

The main project should focus entirely on LLM inference optimization.

Reasons:

- LLM inference already provides sufficient technical depth
- Vision introduces another workload class
- Vision adds different runtime behavior
- Vision adds additional evaluation requirements
- It weakens the core product narrative

If desired, ViT compression can become a separate project later.

---

## Demote the Triton Kernel From Primary Goal to Diagnostic Tool

The custom Triton W4A16 fused dequant-GEMV kernel should remain, but it should not exist simply to demonstrate kernel programming.

Instead, it should support an engineering story like:

```text
Performance model predicted configuration X should win
            ↓
Actual benchmark disagreed
            ↓
Profiler identified dequantization bottleneck
            ↓
Custom Triton kernel implemented
            ↓
Bottleneck reduced
            ↓
Performance model updated
```

This makes the kernel work part of solving an actual system problem.

---

# Suggested CLI Surface

## Hardware Inspection

```bash
quantserve hardware
```

## Trace Analysis

```bash
quantserve analyze traces/workload.jsonl
```

## Benchmark

```bash
quantserve benchmark \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --config configs/search.yaml
```

## Recommendation

```bash
quantserve recommend \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --trace traces/workload.jsonl \
  --p95-ttft 750ms \
  --p95-tpot 50ms \
  --quality-retention 0.98
```

## Regression Gate

```bash
quantserve gate \
  --baseline production.yaml \
  --candidate candidate.yaml
```

## Export

```bash
quantserve export \
  --recommendation recommendation.yaml \
  --format vllm
```

---

# Proposed Optimization Objective

The general optimization problem can be expressed as:

\[
\min_x Cost(x)
\]

subject to:

\[
TTFT_{P95}(x) \le T_{TTFT}
\]

\[
TPOT_{P95}(x) \le T_{TPOT}
\]

\[
Quality(x) \ge Q_{min}
\]

\[
VRAM(x) \le VRAM_{available}
\]

where configuration \(x\) contains:

- Quantization method
- KV-cache precision
- batch configuration
- concurrency
- context length
- scheduler settings
- runtime options

Other objectives may include:

- maximize throughput
- minimize latency
- minimize power
- maximize quality
- minimize cost/token

---

# Evaluation Methodology

The project should evaluate four major questions.

## 1. Prediction Accuracy

How accurately can QuantServe predict:

- TTFT
- TPOT
- throughput
- VRAM usage
- power

Metrics:

- MAE
- RMSE
- MAPE
- R²

---

## 2. Search Efficiency

How quickly can the optimizer find a good configuration?

Metrics:

- Number of benchmarks required
- Fraction of configuration space explored
- GPU-hours consumed
- Regret relative to exhaustive search
- Time-to-best-configuration

---

## 3. Recommendation Quality

Does QuantServe return configurations that satisfy:

- latency SLO
- memory limit
- quality requirement
- cost objective

Metrics:

- SLO satisfaction rate
- recommendation success rate
- deployment regret

---

## 4. Cross-Hardware Generalization

How well does the predictor transfer between GPUs?

Example experiment:

```text
Train on:
RTX 4050
RTX 4090
A10

Test on:
L4
```

Possible approaches:

- Hardware feature encoding
- transfer learning
- hardware-normalized features
- Roofline-derived features

---

# Development Roadmap

## Phase 0 — Foundation

Build:

- CLI
- configuration system
- hardware detection
- baseline vLLM runner
- TTFT/TPOT/throughput metrics
- local GPU baseline

Deliverable:

```text
quantserve benchmark
```

---

## Phase 1 — Benchmark Matrix

Implement:

- FP16
- FP8
- AWQ
- GPTQ
- KV-cache variants
- concurrency sweep
- batching sweep

Collect structured benchmark data.

Deliverable:

```text
benchmarks.parquet
```

---

## Phase 2 — Performance Prediction

Train models predicting:

- TTFT
- TPOT
- throughput
- VRAM

Add:

- train/test split
- cross-validation
- feature importance
- uncertainty intervals

Deliverable:

```text
quantserve predict
```

---

## Phase 3 — Optimizer

Implement:

- configuration space
- constraints
- Bayesian/Optuna search
- surrogate-based candidate selection

Compare against:

- random search
- grid search
- exhaustive search

Deliverable:

```text
quantserve recommend
```

---

## Phase 4 — Quality Gate

Implement:

- perplexity
- reasoning benchmarks
- coding benchmark
- long-context test
- custom evaluator

Integrate quality into optimization constraints.

---

## Phase 5 — Production Traces

Implement:

- metadata-only request capture
- trace parser
- workload fingerprinting
- replay

Deliverable:

```text
quantserve analyze
```

---

## Phase 6 — Explanation Engine

Implement:

- Roofline analysis
- operational intensity
- crossover concurrency
- profiler summaries
- recommendation explanations

Deliverable:

```text
Why this configuration?
```

---

## Phase 7 — Deployment Export

Generate:

- vLLM command
- YAML
- Docker configuration
- Kubernetes manifest

Deliverable:

```text
quantserve export
```

---

## Phase 8 — CI/CD Gate

Implement:

- baseline comparison
- candidate comparison
- regression thresholds
- GitHub Actions integration

Deliverable:

```text
quantserve gate
```

---

## Phase 9 — Drift Detection

Implement:

- workload distribution tracking
- sequence-length drift
- request-rate drift
- concurrency drift
- reoptimization recommendation

---

## Phase 10 — Triton Optimization

Use profiling to identify a genuine bottleneck.

Only then optimize with:

- fused dequantization
- custom GEMV/GEMM kernel
- memory-layout improvements

Benchmark against existing kernels.

---

# Final Project Narrative

The strongest way to describe the finished system is:

> **QuantServe is a hardware- and workload-aware LLM deployment optimizer that learns performance characteristics from a small number of GPU benchmarks and recommends quantization, KV-cache, batching, and serving configurations that satisfy latency, memory, quality, and cost constraints.**

A stronger research-oriented version:

> **QuantServe investigates whether learned surrogate performance models and hardware-aware search can replace exhaustive LLM serving benchmarks while still discovering near-optimal deployment configurations across changing workloads and GPU constraints.**

---

# Portfolio Value

The upgraded project demonstrates:

## Machine Learning

- Regression
- Feature engineering
- Uncertainty estimation
- Surrogate modeling
- Active search

## Optimization

- Bayesian optimization
- Multi-objective optimization
- Constraint satisfaction
- Pareto analysis

## ML Systems

- LLM inference
- vLLM
- Continuous batching
- KV cache
- Quantization
- SLOs

## GPU Engineering

- CUDA behavior
- Memory bandwidth
- Tensor Cores
- Roofline analysis
- Profiling
- Triton

## Production Engineering

- CLI tooling
- Workload traces
- CI/CD
- Deployment generation
- Drift detection
- Regression testing

---

# Final Goal

The project should no longer answer:

> **Which quantization format is faster?**

It should answer:

> **Given this model, this hardware, this workload, this quality floor, and this latency requirement, what configuration should I deploy—and why?**

That should be the core principle guiding every implementation decision in the project.
