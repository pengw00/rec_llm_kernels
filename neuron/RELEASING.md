# Releasing to PyPI

This repo is a monorepo. The Neuron backend is published as a separate Python distribution:

- dist name: `rec-llm-kernels-neuron`
- import package: `rec_llm_kernels_neuron`

## 1) Bump version

Edit `neuron/pyproject.toml` and update `project.version`.

## 2) Build

From the repo root:
```bash
python -m pip install -U build twine
python -m build neuron
python -m twine check dist/*
```

## 3) Upload

TestPyPI (recommended first):
```bash
python -m twine upload -r testpypi dist/*
```

PyPI:
```bash
python -m twine upload dist/*
```

