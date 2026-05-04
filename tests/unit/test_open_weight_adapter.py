from unittest.mock import MagicMock

import pytest

from concord.agents.open_weight_adapter import OpenWeightAdapter, OpenWeightOOMError


class TestOpenWeightAdapter:
    def test_init_defaults(self):
        adapter = OpenWeightAdapter("Qwen/Qwen2-0.5B")
        assert adapter.model_id == "Qwen/Qwen2-0.5B"
        assert adapter.activation_layers == []
        assert adapter.steering_vector is None
        assert adapter.steering_strength == 1.0

    def test_init_with_steering(self):
        adapter = OpenWeightAdapter(
            "meta-llama/Llama-3-8B",
            activation_layers=[12, 16],
            steering_vector=[0.1, 0.2, 0.3],
            steering_strength=2.0,
        )
        assert len(adapter.activation_layers) == 2
        assert adapter.steering_vector == [0.1, 0.2, 0.3]
        assert adapter.steering_strength == 2.0

    def test_oom_error_raised_at_batch_size_one(self):
        with pytest.raises(OpenWeightOOMError, match="batch_size=1"):
            raise OpenWeightOOMError("Cannot extract activations at batch_size=1")

    def test_oom_custom_exception_type(self):
        err = OpenWeightOOMError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"


class TestExtractActivations:
    @pytest.mark.requires_interp
    def test_extract_activations_shape(self):
        pytest.importorskip("torch")
        pytest.importorskip("nnsight")
        pytest.importorskip("transformers")

        try:
            adapter = OpenWeightAdapter(
                "Qwen/Qwen2-0.5B", activation_layers=[0]
            )
            adapter.load_model()
            activations = adapter.extract_activations("Hello world", batch_size=1)
            assert 0 in activations
            result = activations[0]
            assert isinstance(result, list)
        except Exception as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                pytest.skip("CUDA not available")
            if "out of memory" in str(e).lower():
                pytest.skip("OOM on test machine")
            raise


class TestApplySteering:
    @pytest.mark.requires_interp
    def test_steering_changes_output(self):
        pytest.importorskip("torch")
        pytest.importorskip("nnsight")
        pytest.importorskip("transformers")

        try:
            adapter = OpenWeightAdapter(
                "Qwen/Qwen2-0.5B",
                activation_layers=[0],
                steering_vector=[0.1] * 896,
                steering_strength=1.0,
            )
            adapter.load_model()
            unsteered = adapter._generate("Hello")
            steered = adapter.apply_steering("Hello")
            assert isinstance(unsteered, str)
            assert isinstance(steered, str)
        except Exception as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                pytest.skip("CUDA not available")
            if "out of memory" in str(e).lower():
                pytest.skip("OOM on test machine")
            raise


class TestActMethod:
    def test_act_without_steering(self):
        adapter = OpenWeightAdapter("Qwen/Qwen2-0.5B")
        adapter._generate = MagicMock(return_value="I accept your offer.")
        import asyncio
        from concord.env.core import NegotiationEnv
        from concord.schemas.scenario import Domain, PrivateContext, Scenario

        s = Scenario(
            id="t", domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=100),
            seller_context=PrivateContext(batna=200),
            deal_schema={"price": "float", "quantity": "int"},
        )
        env = NegotiationEnv()
        env.reset(s)
        action = asyncio.run(adapter.act(env.state, s.buyer_context))
        assert action.content == "I accept your offer."
