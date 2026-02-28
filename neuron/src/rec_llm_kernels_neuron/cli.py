from __future__ import annotations

import sys


def run_tests() -> None:
    """
    Run the unit tests shipped with this package.

    Usage:
      rec-llm-kernels-neuron-test [-q] [pytest args...]

    This requires `pytest` to be installed in the current environment.
    """
    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is not installed. Run: pip install -U pytest", file=sys.stderr)
        raise SystemExit(2)

    # `--pyargs` lets pytest discover tests from an installed package.
    args = ["--pyargs", "rec_llm_kernels_neuron.tests"]
    args.extend(sys.argv[1:])
    raise SystemExit(pytest.main(args))

