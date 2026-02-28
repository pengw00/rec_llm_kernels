import pytest


torch = pytest.importorskip("torch")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for these kernel tests.")


def test_extension_import_and_symbols():
    _require_cuda()
    _C = pytest.importorskip("rec_llm_kernels._C")

    assert hasattr(_C, "ops"), "Expected rec_llm_kernels._C.ops submodule"
    for name in ("flash_att_forward", "reshape_and_cache", "rms_norm"):
        assert hasattr(_C.ops, name), f"Expected rec_llm_kernels._C.ops.{name}"

