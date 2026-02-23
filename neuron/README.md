# `rec-llm-kernels-neuron`

This is a standalone (CPU/XLA/Neuron) backend package intended to be installed on **AWS Inf2** without
triggering the CUDA/CMake build from the main repo.

## Install (Inf2)

From the repo root:

```bash
pip install -e neuron
```

## Run unit tests (Inf2)

```bash
pytest -q tests_inf2
```

