# Contributing to QuantServe

QuantServe is an open source prototype of a hardware and workload aware LLM
deployment optimizer. Contributions that improve measured serving evidence,
correctness, reproducibility, and documentation are welcome.

## Local setup

Use Python 3.10 or newer in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make test PYTHON=python
```

The default test suite runs on CPU. GPU and Triton checks run when compatible
hardware and software are available. CI tests Python 3.10 and 3.12 and builds
the source and wheel distributions.
Vercel preview builds are skipped because this repository has no web app or
Python web entrypoint; GitHub Actions provides the package checks.

## Changes and evidence

- Keep pull requests focused, and describe the behavior, tests, and remaining
  limitations in the PR body.
- Label simulated, predicted, derived, and measured values clearly. A mock
  benchmark or synthetic quality task is not evidence of GPU serving behavior.
- For measured results, include the model revision, serving engine version,
  hardware, precision, workload definition, run command, and raw result file.
- Avoid adding model weights, private request text, credentials, or generated
  benchmark output to the repository.

Start with an issue for larger changes so the measurement contract and scope
can be agreed on before implementation. Small fixes can go directly to a PR.
