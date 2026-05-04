import pytest


@pytest.mark.requires_interp
@pytest.mark.slow
class TestOpenWeightIntegration:
    def test_load_small_model(self):
        pytest.importorskip("torch")
        pytest.importorskip("nnsight")
        pytest.importorskip("transformers")

        from concord.agents.open_weight_adapter import OpenWeightAdapter

        try:
            adapter = OpenWeightAdapter("Qwen/Qwen2-0.5B")
            adapter.load_model()
            assert adapter._model is not None
            assert adapter._tokenizer is not None
        except Exception as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                pytest.skip("CUDA not available")
            raise

    def test_generate_text(self):
        pytest.importorskip("torch")
        pytest.importorskip("nnsight")
        pytest.importorskip("transformers")

        from concord.agents.open_weight_adapter import OpenWeightAdapter

        try:
            adapter = OpenWeightAdapter("Qwen/Qwen2-0.5B")
            adapter.load_model()
            output = adapter._generate("Hello, my name is")
            assert isinstance(output, str)
            assert len(output) > 0
        except Exception as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                pytest.skip("CUDA not available")
            if "out of memory" in str(e).lower():
                pytest.skip("OOM on test machine")
            raise
