"""LLM-powered director using Qwen2.5 via Hugging Face transformers."""

import logging

from stage_director.schemas import DirectResponse, LightingPreset
from stage_director.services.parsing import DirectorOutputError, parse_llm_output

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a stage director assistant controlling a 3D stage.
Reply with ONLY one JSON object and nothing else, in this exact format:
{"dialogue": "<short spoken reply>", "lighting_preset": "<preset>"}
Allowed lighting_preset values: neutral, red_alert, blue_calm, spotlight.
Use neutral if the command does not ask for a lighting change."""


class LLMDirector:
    """Loads the model once, then converts commands to validated JSON."""

    def __init__(self, model_id: str, max_new_tokens: int = 100) -> None:
        # Heavy imports live here so laptops without these libraries still run.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._max_new_tokens = max_new_tokens
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info("Loading LLM %s ...", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        logger.info("LLM loaded")

    def generate_raw(self, command: str) -> str:
        """Return the model's raw text for a command (no parsing, no fallback)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def direct(self, command: str) -> DirectResponse:
        logger.info("LLM received command: %s", command)
        raw = self.generate_raw(command)
        logger.debug("Raw LLM output: %s", raw)

        try:
            return parse_llm_output(raw, command)
        except DirectorOutputError:
            logger.warning("Falling back to neutral. Raw output: %r", raw)
            return DirectResponse(
                received_command=command,
                dialogue="Sorry, I did not understand that command.",
                lighting_preset=LightingPreset.NEUTRAL,
            )