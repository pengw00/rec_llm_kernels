# CUDA package

This folder contains the NVIDIA CUDA extension build (`CMakeLists.txt`, `setup.py`, `csrc/`) and the
Python package wrapper (`python/rec_llm_kernels/`).

Install (CUDA machine):
```bash
pip install -e cuda
```


2. run cuda test in GPU g5.xlarge
```
cd opt/dlami/nvme
clone rec_llm_kernels
cd rec_llm_kernels
python3 -m venv venv
source venv/bin/activate
export TMPDIR = opt/dlami/nvme
mkdir -p $TMPDIR
pip install --no-cache-dir torch
pip install --no-cache-dir --no-build-isolation -e cuda
```
