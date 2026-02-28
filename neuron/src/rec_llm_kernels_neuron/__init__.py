"""
Inferentia2 (AWS Neuron) backend (reference implementation).

This package is intended to be installed on Inf2 without triggering any CUDA/CMake builds.
"""

from .ops import flash_att_forward, paged_attention_decode, reshape_and_cache, rms_norm  # noqa: F401
