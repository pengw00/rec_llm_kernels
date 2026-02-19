import pytest


torch = pytest.importorskip("torch")


def test_llama_layer_placeholder():
    pytest.skip(
        "rec_llm.model_executor Llama layer and PagedAttention are not implemented/consistent yet; "
        "enable this test once the Python model executor and C++ bindings are wired end-to-end."
    )

