# `rec-llm-kernels-neuron`

This is a standalone (CPU/XLA/Neuron) backend package intended to be installed on **AWS Inf2** without
triggering the CUDA/CMake build from the main repo.

## Install (Inf2)

From the repo root:

```bash
pip install -e neuron
```

If you need to publish this package to PyPI, see `RELEASING.md`.

## Run unit tests (Inf2)

```bash
pytest -q tests_inf2
```

## Run unit tests on EC2 Inf2 (install from GitHub)

On an Inf2 instance, you can install just this subpackage directly from GitHub (no CUDA/CMake build):

```bash
pip install "git+https://github.com/<YOUR_GITHUB>/<YOUR_REPO>.git@<TAG_OR_COMMIT>#subdirectory=neuron"
pip install -U pytest
rec-llm-kernels-neuron-test -q
```

Alternative (pure pytest):
```bash
pytest -q --pyargs rec_llm_kernels_neuron.tests
```
