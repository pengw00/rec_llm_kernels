# AWS Inf2 (Inferentia2 / Neuron) — Minimal Backend + Unit Tests

You asked for a **simple Neuron backend + “kernel” path** that lets unit tests run on **Inf2**,
without touching the existing CUDA extension code path.

This repo now includes an **additive** Neuron backend package:

- `rec_llm_kernels_neuron/` (pure PyTorch/XLA reference ops; runs on Neuron via `torch-neuronx`)
- `tests_inf2/` (unit tests that require `torch_xla` and run on the XLA device)

## On an Inf2 instance

1) Install AWS Neuron + torch-neuronx (so `import torch_xla` works)

2) Clone repo and install the Neuron-only package (does **not** build CUDA)
```bash
git clone <YOUR_REPO_URL>
cd rec_llm_kernels
pip install -e neuron
```

3) Run tests
```bash
pytest -q tests_inf2
```

## Notes

- Today this is a correctness-first backend (pure PyTorch ops). Next step is to replace individual ops with
  Neuron Custom C++ Ops or NKI kernels for performance.
- CUDA tests remain unchanged and continue to target `rec_llm_kernels._C`.
