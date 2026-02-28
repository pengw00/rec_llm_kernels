from __future__ import annotations

from pathlib import Path

import rec_llm_kernels_neuron


def tests_path() -> str:
    """
    Return the filesystem path to the installed unit tests shipped with
    `rec_llm_kernels_neuron`.
    """
    pkg_dir = Path(rec_llm_kernels_neuron.__file__).resolve().parent
    return str(pkg_dir / "tests")

