import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from transformers import AutoTokenizer

from source.module.generate.base import BaseGenerator, BaseGeneratorConfig


@dataclass
class VLLMServerGeneratorConfig(BaseGeneratorConfig):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    served_model_name: Optional[str] = None
    base_url: str = "http://127.0.0.1:8000/v1"
    max_total_tokens: int = 4096
    max_new_tokens: int = 64
    min_new_tokens: int = 1
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)
    request_timeout: float = 600.0


class VLLMServerGenerator(BaseGenerator):
    """OpenAI-compatible vLLM server client for generation-only inference."""

    def __init__(self, cfg: VLLMServerGeneratorConfig):
        super().__init__(cfg)
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.served_model_name or cfg.model_name
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
        self.tokenizer.padding_side = "left"
        self._health_check_completion()

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            print(
                "[vllm-server-error] "
                f"url={url} status={error.code} reason={error.reason} "
                f"payload={self._summarize_payload(payload)}",
                file=sys.stderr,
            )
            if body:
                print(f"[vllm-server-error-body] {body[:4000]}", file=sys.stderr)
            raise RuntimeError(f"vLLM server request failed: {error.code} {body}") from error
        except urllib.error.URLError as error:
            print(f"[vllm-server-error] url={url} connection_error={error}", file=sys.stderr)
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                "Start it first with script/preload_vllm_server.py."
            ) from error

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            prompt_count = len(prompt)
            prompt_chars = sum(len(str(item)) for item in prompt)
            sample_count = min(4, prompt_count)
            sample_tokens = sum(
                len(self.tokenizer.encode(str(item), add_special_tokens=False))
                for item in prompt[:sample_count]
            )
            prompt_tokens_estimate = int(sample_tokens * prompt_count / sample_count) if sample_count else 0
        elif prompt is None:
            prompt_count = 0
            prompt_chars = 0
            prompt_tokens_estimate = 0
        else:
            prompt_count = 1
            prompt_chars = len(str(prompt))
            prompt_tokens_estimate = len(self.tokenizer.encode(str(prompt), add_special_tokens=False))

        return {
            "model": payload.get("model"),
            "prompt_count": prompt_count,
            "prompt_chars": prompt_chars,
            "prompt_tokens_estimate": prompt_tokens_estimate,
            "max_tokens": payload.get("max_tokens"),
        }

    def _health_check_completion(self):
        payload = {
            "model": self.model,
            "prompt": "health check",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        self._post_json("/completions", payload)

    @staticmethod
    def _choice_texts_by_index(response: Dict[str, Any], expected_count: int) -> List[str]:
        outputs = [""] * expected_count
        for fallback_idx, choice in enumerate(response.get("choices", [])):
            index = int(choice.get("index", fallback_idx))
            if 0 <= index < expected_count:
                outputs[index] = choice.get("text", "")
        return outputs

    def _generate(self, inputs: List[str]) -> List[str]:
        stop = [self.tokenizer.eos_token, "<|eot_id|>"] + list(self.cfg.stop)
        stop = [item for item in stop if item]
        max_prompt_tokens = max(1, int(self.cfg.max_total_tokens) - int(self.cfg.max_new_tokens))
        truncated_inputs = [
            self.tokenizer.decode(
                self.tokenizer.encode(
                    input_text,
                    max_length=max_prompt_tokens,
                    truncation=True,
                    add_special_tokens=False,
                ),
                skip_special_tokens=False,
            )
            for input_text in inputs
        ]
        payload = {
            "model": self.model,
            "prompt": truncated_inputs,
            "n": 1,
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "stop": stop,
            "stream": False,
        }
        response = self._post_json("/completions", payload)
        return self._choice_texts_by_index(response, len(inputs))

    def _score(
        self,
        input_texts: List[str],
        output_texts: List[str],
        method: Literal["perplexity_score"] = "perplexity_score",
    ):
        raise NotImplementedError("VLLMServerGenerator is generation-only for inference.")
