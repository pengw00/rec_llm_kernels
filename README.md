# rec_llm_kernels

## Project Structure
- `cuda/`: CUDA extension build + Python wrapper (`rec_llm_kernels`).
  - `cuda/csrc/`: C++/CUDA source code.
  - `cuda/python/rec_llm_kernels/`: Python package wrapper (imports `_C.so` after build).
- `runtime/`: Minimal inference runtime scaffolding (`rec_llm_runtime`).
- `neuron/`: Neuron/Inf2 backend package (`rec_llm_kernels_neuron`).
- `docs/`: Roadmap and Inf2 notes.

## Build & Install (CUDA)
This project contains CUDA (`.cu`) sources and generally needs **Linux + NVIDIA GPU + CUDA toolkit** to build.

### Create and activate venv
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip setuptools wheel ninja
```

### Install core dependencies
```bash
python3 -m pip install torch
```

### Build and install the CUDA extension
```bash
python3 -m pip install -e cuda -v
```

### (Recommended) Install the runtime package
Some tests and scripts import `rec_llm_runtime`:
```bash
python3 -m pip install -e runtime
```

### Recompiling C++/CUDA Kernels
If you modify any files in `cuda/csrc/` (`.cu` or `.cpp`), recompile the binary:

Option A: Fast in-place build (recommended for debugging).  
This generates the `.so` binary directly in your project folder for immediate testing.
```bash
python3 cuda/setup.py build_ext --inplace
```

Option B: Clean build (recommended for production).
```bash
MAX_JOBS=4 python3 -m pip install -e cuda -v
```

## EC2 (G5 / A10G) Build + Full Test Run
This is a practical end-to-end flow for an EC2 `g5.*` instance (A10G, `sm86`).

### Using the local SSD / NVMe (recommended)
Many DLAMIs mount the instance-store NVMe under `/opt/dlami/nvme`. Use it for the repo checkout and for caches,
otherwise you may hit `No space left on device` under `/home/ubuntu`.

Check disk mounts:
```bash
df -h
df -h /opt/dlami/nvme || true
```

#### Pull / checkout the repo onto NVMe
Clone directly onto the NVMe mount:
```bash
cd /opt/dlami/nvme
git clone <YOUR_REPO_URL> rec_llm_kernels
cd rec_llm_kernels
```

If you already have a checkout elsewhere, copy it to NVMe (example):
```bash
rsync -a --delete --exclude venv /home/ubuntu/rec_llm_kernels/ /opt/dlami/nvme/rec_llm_kernels/
cd /opt/dlami/nvme/rec_llm_kernels
```

Suggested cache locations on the NVMe:
```bash
export XDG_CACHE_HOME=/opt/dlami/nvme/.cache
export FLASHINFER_WORKSPACE_DIR=/opt/dlami/nvme/.cache/flashinfer
export TORCH_EXTENSIONS_DIR=/opt/dlami/nvme/.cache/torch_extensions
export TMPDIR=/opt/dlami/nvme/tmp
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
```

### 0) Sanity check (GPU + PyTorch)
```bash
nvidia-smi
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"
```

### 1) Create and activate venv
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip setuptools wheel ninja pytest
```

### 2) Install the runtime package (pure Python)
`tests_cuda/test_llama_layer.py` imports `rec_llm_runtime`, so install it first:
```bash
python3 -m pip install -e runtime
```

### 3) Build + install the CUDA extension (editable)
For A10G on G5, set `sm86`:
```bash
export CMAKE_CUDA_ARCHITECTURES=86
export MAX_JOBS=4
python3 -m pip install -e cuda -v
```

### 4) Run all tests
```bash
python3 -m pytest -ra tests_cuda
python3 -m pytest -ra tests_contract
```

### 5) (Optional) Enable FlashInfer v2 decode tests
FlashInfer is an optional fast-path used by `paged_attention_decode_v2_*`.

Install:
```bash
python3 -m pip install -U flashinfer-python
```

If your home volume is small (common on AMIs), point FlashInfer's cache/workspace to a larger disk (e.g. NVMe):
```bash
export XDG_CACHE_HOME=/opt/dlami/nvme/.cache
export FLASHINFER_WORKSPACE_DIR=/opt/dlami/nvme/.cache/flashinfer
mkdir -p "$FLASHINFER_WORKSPACE_DIR"
python3 -c "import flashinfer; print('flashinfer ok', flashinfer.__version__)"
```

Run the FlashInfer-specific test:
```bash
python3 -m pytest -q tests_cuda/test_paged_attention_decode_v2_flashinfer.py -rs
```

### FlashInfer (optional CMake integration)
If you want to include FlashInfer headers in the CUDA extension build (for C++/CUDA kernels), CMake will **not**
fetch it over the network by default. Choose one:

- Use a local checkout:
```bash
USE_FLASHINFER=1 python3 -m pip install -e cuda --config-settings=cmake.args="-DFLASHINFER_SOURCE_DIR=/path/to/flashinfer"
```

- Allow CMake to fetch it (requires network access):
```bash
USE_FLASHINFER=1 python3 -m pip install -e cuda --config-settings=cmake.args="-DFLASHINFER_FETCH=ON"
```

## Testing (pytest)
Kernel tests require CUDA. On non-CUDA machines, tests will be skipped.

```bash
python3 -m pip install pytest
python3 -m pytest -q tests_cuda
```

To show skip reasons in the output:
```bash
python3 -m pytest -q -rs tests_cuda
```

To show a short test summary including skipped/failed/xfailed reasons:
```bash
python3 -m pytest -ra tests_cuda
```

### Troubleshooting
- `CUDA error: invalid configuration argument` in `reshape_and_cache`: older kernel launch used a 2D thread block
  (`head_dim` x `num_heads`) which can exceed 1024 threads/block (e.g. `32 * 128 = 4096`). Rebuild after pulling
  the fix and the kernel will launch as one block per `(token, head)`.
- Import error with an undefined symbol (example: `c10::cuda::c10_cuda_check_implementation`): your extension was
  built/linked against a different Torch build, or is missing Torch CUDA link targets. Do a clean rebuild:
  ```bash
  cd /opt/dlami/nvme/rec_llm_kernels
  source venv/bin/activate
  rm -rf cuda/build build *.egg-info **/__pycache__
  python3 -m pip install -e cuda -v
  python3 -c "import rec_llm_kernels._C as _C; print('import ok')"
  python3 -m pytest -ra tests_cuda
  ```
- FlashInfer import fails with `OSError: [Errno 28] No space left on device`: set `XDG_CACHE_HOME` and
  `FLASHINFER_WORKSPACE_DIR` to a larger filesystem (e.g. `/opt/dlami/nvme/.cache`), then re-import.

### Colab rebuild (if extension import fails)
If you hit an error like `undefined symbol ... type_caster<at::Tensor>::load(...)` when importing `rec_llm_kernels._C`,
do a clean rebuild:

```bash
cd /content/rec_llm_kernels
rm -rf build
rm -rf *.egg-info
rm -rf **/__pycache__

pip uninstall -y rec-llm-kernels rec_llm_kernels || true
MAX_JOBS=2 pip install -e cuda -v
pytest -q tests_cuda
```

### Colab quickstart
In Colab, set Runtime → Change runtime type → GPU, then run:
```bash
git clone <YOUR_REPO_URL> rec_llm_kernels
cd rec_llm_kernels
pip -q install --upgrade pip
pip -q install torch pytest ninja setuptools wheel
pip -q install -e cuda
pytest -q tests_cuda
```

### Tiny Llama smoke test (Transformers)
Quick sanity check that a tiny Llama model can run on GPU and returns `past_key_values`:
```bash
pip -q install transformers accelerate sentencepiece
python scripts/colab_tiny_llama_smoke.py
```

## Kubernetes (model serving + training simulation)
This repo also includes a minimal Kubernetes-based model serving + training simulation system:

- **Inference Server Cluster**: 2 replicas behind a Service (load balanced)
  - Prompt processing: `POST /generate`
  - Model version tracking: response header `X-Model-Version`
  - Zero-downtime model updates: hot reload in-place (no pod restart) via mounted `ConfigMap` + `POST /reload`
- **Fake Trainer Server**:
  - Sends prompt requests to the inference Service
  - Emits model "weight" updates by patching the `ConfigMap` and triggering reload on all inference pods

See `/k8s/README.md` for minikube instructions and a quick demo.

## AWS Inf2 (Neuron)
See `/docs/INF2.md` for running the Neuron backend tests on Inf2.
