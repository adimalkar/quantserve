PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help install test bench profile eval pareto micro-kernel download-models demos clean

help:
	@echo "QuantServe Command Suite:"
	@echo "  make test             Run unit and integration tests"
	@echo "  make demos            Run all 3 primary deployment optimization demos"
	@echo "  make bench            Execute open-loop load benchmark across precisions"
	@echo "  make profile          Run PyTorch Kineto kernel breakdown and roofline analysis"
	@echo "  make eval             Execute multi-task quality evaluation with bootstrap CIs"
	@echo "  make micro-kernel     Run Triton fused dequant-GEMV micro-benchmarks"
	@echo "  make pareto           Generate Pareto frontier and crossover plots"
	@echo "  make download-models  Download optional project model weights"
	@echo "  make clean            Remove build artifacts and temporary run logs"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=. $(PYTEST) -v tests/

bench:
	PYTHONPATH=. $(PYTHON) -m src.bench.runner --config configs/workload_fast.yaml

profile:
	PYTHONPATH=. $(PYTHON) -m src.analysis.roofline --device ada_4050
	PYTHONPATH=. $(PYTHON) -m src.profiling.profiler --batch-sizes 1 4 16 32 64

eval:
	PYTHONPATH=. $(PYTHON) -m src.eval.eval_harness --model llama3-1b --precisions fp16 int4-awq w8a8 fp8

micro-kernel:
	PYTHONPATH=. $(PYTHON) -m src.internals.bench_kernel

pareto:
	PYTHONPATH=. $(PYTHON) -m src.viz.plot_pareto
	PYTHONPATH=. $(PYTHON) -m src.viz.plot_crossover
	PYTHONPATH=. $(PYTHON) -m src.viz.plot_roofline

download-models:
	PYTHONPATH=. $(PYTHON) scripts/download_models.py

demos:
	@echo "Running Demo 1: Interactive Coding Assistant..."
	PYTHONPATH=. $(PYTHON) examples/coding_assistant/run_demo.py
	@echo "Running Demo 2: RAG Knowledge Assistant..."
	PYTHONPATH=. $(PYTHON) examples/rag_assistant/run_demo.py
	@echo "Running Demo 3: Batch Document Processing..."
	PYTHONPATH=. $(PYTHON) examples/batch_processing/run_demo.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache outputs/ __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
