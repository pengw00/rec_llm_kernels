"""
Inferentia2 (AWS Neuron) backend.

This package is intentionally separate from `rec_llm_kernels` so it can be used
on Inf2 instances without importing the CUDA extension package.
"""

from .ops import paged_attention_decode, reshape_and_cache, rms_norm  # noqa: F401

