from concord.agents.base import Action, AgentProtocol
from concord.env.offer_parser import parse_offer as parse_raw
from concord.exceptions import ConcordError
from concord.schemas.episode import ActionType


class OpenWeightOOMError(ConcordError):
    pass


class OpenWeightAdapter(AgentProtocol):
    def __init__(
        self,
        model_id: str,
        activation_layers: list[int] | None = None,
        steering_vector: list[float] | None = None,
        steering_strength: float = 1.0,
    ):
        self.model_id = model_id
        self.activation_layers = activation_layers or []
        self.steering_vector = steering_vector
        self.steering_strength = steering_strength
        self._model = None
        self._tokenizer = None
        self._activation_cache: dict[int, list] = {}

    def load_model(self) -> None:
        try:
            import nnsight
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Open-weight adapter requires the [interp] extra. "
                "Install with: pip install concord-bench[interp]"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._activation_cache = {}

    def extract_activations(self, prompt: str, batch_size: int = 1) -> dict[int, list]:
        try:
            import nnsight
            import torch
        except ImportError:
            raise ImportError(
                "Open-weight adapter requires the [interp] extra. "
                "Install with: pip install concord-bench[interp]"
            )

        if self._model is None:
            self.load_model()

        def _extract(current_batch: int) -> dict[int, list]:
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(
                    next(self._model.parameters()).device
                )
                activations: dict[int, list] = {}
                with torch.no_grad():
                    with nnsight.trace(self._model) as tracer:
                        with tracer.invoke(inputs.input_ids) as invoker:
                            for layer_idx in self.activation_layers:
                                layer_output = self._model.model.layers[
                                    layer_idx
                                ].output[0]
                                activations[layer_idx] = layer_output.save()
                return {k: v.value.cpu().tolist() for k, v in activations.items()}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch <= 1:
                    raise OpenWeightOOMError(
                        "Cannot extract activations at batch_size=1"
                    )
                return _extract(current_batch // 2)

        return _extract(batch_size)

    def apply_steering(self, prompt: str) -> str:
        try:
            import nnsight
            import torch
        except ImportError:
            raise ImportError(
                "Open-weight adapter requires the [interp] extra. "
                "Install with: pip install concord-bench[interp]"
            )

        if self._model is None:
            self.load_model()
        if self.steering_vector is None:
            return self._generate(prompt)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            next(self._model.parameters()).device
        )
        steering_tensor = torch.tensor(
            self.steering_vector, dtype=torch.float16
        ).to(next(self._model.parameters()).device)

        with torch.no_grad():
            with nnsight.trace(self._model) as tracer:
                with tracer.invoke(inputs.input_ids) as invoker:
                    for layer_idx in self.activation_layers:
                        hidden = self._model.model.layers[layer_idx].output[0]
                        hidden = hidden + self.steering_strength * steering_tensor
                output = invoker.output.logits

        generated_ids = output.argmax(dim=-1)[:, -1:]
        return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _generate(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            next(self._model.parameters()).device
        )
        with torch.no_grad():
            output = self._model.generate(inputs.input_ids, max_new_tokens=128)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    async def act(self, env_state, private_ctx) -> Action:
        scenario = env_state.scenario
        turns = env_state.turns
        my_role = env_state.current_agent
        private = private_ctx

        transcript = ""
        for t in turns:
            agent_label = "You" if t.agent == my_role else "Counterparty"
            transcript += f"{agent_label}: {t.content}\n"

        prompt = f"""You are the {my_role} in a {scenario.domain.value} negotiation.
Scenario: {scenario.scenario_description}
Your BATNA: ${private.batna}
Reserve price: ${private.reserve_price or 'unspecified'}
Hard constraints: {', '.join(private.hard_constraints) or 'none'}

Transcript:
{transcript if transcript else 'No messages yet.'}

Respond with your negotiation message and optionally a JSON offer."""

        if self.steering_vector is not None and self.activation_layers:
            raw_output = self.apply_steering(prompt)
        else:
            raw_output = self._generate(prompt)

        offer_dict = None
        action_type = ActionType.MESSAGE

        import re
        json_match = re.search(r'\{.*"domain".*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                offer_dict = parse_raw(
                    json_match.group(0), scenario.domain.value
                ).model_dump()
                action_type = ActionType.OFFER
            except Exception:
                pass

        if "walk away" in raw_output.lower():
            action_type = ActionType.WALK_AWAY
            offer_dict = None
        elif "accept" in raw_output.lower() and action_type != ActionType.OFFER:
            action_type = ActionType.ACCEPT

        return Action(action_type=action_type, content=raw_output, offer_dict=offer_dict)
