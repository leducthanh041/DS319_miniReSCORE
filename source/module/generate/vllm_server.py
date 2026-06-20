import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from math import exp
from typing import Any, Dict, List, Optional, Literal

from transformers import AutoTokenizer

from source.module.generate.base import BaseGenerator, BaseGeneratorConfig


@dataclass
class VLLMServerGeneratorConfig(BaseGeneratorConfig):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    served_model_name: Optional[str] = None
    base_url: str = "http://127.0.0.1:8000/v1"
    max_total_tokens: int = 4096
    score_max_tokens: int = 1024
    max_new_tokens: int = 64
    min_new_tokens: int = 1
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)
    request_timeout: float = 600.0
    scoring_mode: str = "prompt_logprobs"
    prompt_logprobs: int = 20
    missing_logprob_fallback: float = -20.0
    context_safety_margin: int = 16


class VLLMServerGenerator(BaseGenerator):
    """OpenAI-compatible vLLM server client for generation-only inference."""

    def __init__(self, cfg: VLLMServerGeneratorConfig):
        super().__init__(cfg)
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.served_model_name or cfg.model_name
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
        self.tokenizer.padding_side = "left"
        self._warned_score_fallback = False
        self._warned_incomplete_prompt_logprobs = False
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
            "logprobs": payload.get("logprobs"),
            "prompt_logprobs": payload.get("prompt_logprobs"),
            "echo": payload.get("echo"),
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
        max_tokens = min(
            int(self.cfg.max_new_tokens),
            max(1, int(self.cfg.max_total_tokens) - 1),
        )
        max_prompt_tokens = max(
            1,
            int(self.cfg.max_total_tokens)
            - max_tokens
            - int(self.cfg.context_safety_margin),
        )
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
            "max_tokens": max_tokens,
            "temperature": self.cfg.temperature,
            "stop": stop,
            "stream": False,
        }
        response = self._post_json("/completions", payload)
        return self._choice_texts_by_index(response, len(inputs))

    @staticmethod
    def _extract_logprob(logprob_entry: Any) -> float:
        if isinstance(logprob_entry, dict):
            if "logprob" in logprob_entry:
                return float(logprob_entry["logprob"])
            if "log_prob" in logprob_entry:
                return float(logprob_entry["log_prob"])
        return float(logprob_entry)

    def _score_with_prompt_logprobs(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> List[float]:
        combined_texts = []
        output_token_ids = []
        output_offsets = []
        max_prompt_tokens = max(
            2,
            int(self.cfg.score_max_tokens)
            - 1
            - int(self.cfg.context_safety_margin),
        )

        for input_text, output_text in zip(input_texts, output_texts):
            cur_output_ids = self.tokenizer.encode(
                output_text,
                add_special_tokens=False,
            ) or [self.tokenizer.eos_token_id]
            cur_output_ids = cur_output_ids[: max(1, max_prompt_tokens - 1)]
            max_input_tokens = max(1, max_prompt_tokens - len(cur_output_ids))
            cur_input_ids = self.tokenizer.encode(
                input_text,
                max_length=max_input_tokens,
                truncation=True,
                add_special_tokens=False,
            ) or [self.tokenizer.eos_token_id]

            output_offsets.append(len(cur_input_ids))
            output_token_ids.append(cur_output_ids)
            combined_texts.append(
                self.tokenizer.decode(
                    cur_input_ids + cur_output_ids,
                    skip_special_tokens=False,
                )
            )

        response = self._post_json(
            "/completions",
            {
                "model": self.model,
                "prompt": combined_texts,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": self.cfg.prompt_logprobs,
                "stream": False,
            },
        )
        choices = sorted(
            response.get("choices", []),
            key=lambda choice: int(choice.get("index", 0)),
        )
        if len(choices) < len(combined_texts):
            raise RuntimeError(
                "vLLM server returned fewer scoring choices than requested."
            )

        perplexities = []
        for request_idx, choice in enumerate(choices[:len(combined_texts)]):
            prompt_logprobs = choice.get("prompt_logprobs")
            if not prompt_logprobs:
                raise RuntimeError(
                    "vLLM server response does not contain prompt_logprobs."
                )

            token_logprobs = []
            for token_offset, token_id in enumerate(output_token_ids[request_idx]):
                position = output_offsets[request_idx] + token_offset
                if position >= len(prompt_logprobs):
                    if not self._warned_incomplete_prompt_logprobs:
                        print(
                            "[vllm-server-warning] Incomplete prompt_logprobs; "
                            f"using fallback={self.cfg.missing_logprob_fallback}.",
                            file=sys.stderr,
                        )
                        self._warned_incomplete_prompt_logprobs = True
                    token_logprobs.append(self.cfg.missing_logprob_fallback)
                    continue

                candidates = prompt_logprobs[position] or {}
                entry = candidates.get(str(token_id), candidates.get(token_id))
                if entry is None:
                    if not self._warned_score_fallback:
                        print(
                            "[vllm-server-warning] Target token missing from "
                            "prompt_logprobs; using configured fallback.",
                            file=sys.stderr,
                        )
                        self._warned_score_fallback = True
                    token_logprobs.append(self.cfg.missing_logprob_fallback)
                else:
                    token_logprobs.append(self._extract_logprob(entry))

            mean_nll = -sum(token_logprobs) / max(1, len(token_logprobs))
            perplexities.append(exp(mean_nll))
        return perplexities

    def _score_with_echo_logprobs(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> List[float]:
        combined_texts = []
        input_token_lengths = []
        output_token_lengths = []
        max_prompt_tokens = max(
            2,
            int(self.cfg.score_max_tokens)
            - 1
            - int(self.cfg.context_safety_margin),
        )

        for input_text, output_text in zip(input_texts, output_texts):
            output_ids = self.tokenizer.encode(
                output_text,
                add_special_tokens=False,
            ) or [self.tokenizer.eos_token_id]
            output_ids = output_ids[: max(1, max_prompt_tokens - 1)]
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=max(1, max_prompt_tokens - len(output_ids)),
                truncation=True,
                add_special_tokens=False,
            ) or [self.tokenizer.eos_token_id]
            combined_texts.append(
                self.tokenizer.decode(
                    input_ids + output_ids,
                    skip_special_tokens=False,
                )
            )
            input_token_lengths.append(len(input_ids))
            output_token_lengths.append(len(output_ids))

        response = self._post_json(
            "/completions",
            {
                "model": self.model,
                "prompt": combined_texts,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 1,
                "echo": True,
                "stream": False,
            },
        )
        choices = sorted(
            response.get("choices", []),
            key=lambda choice: int(choice.get("index", 0)),
        )

        perplexities = []
        for request_idx, choice in enumerate(choices[:len(combined_texts)]):
            logprobs = choice.get("logprobs") or {}
            token_logprobs = logprobs.get("token_logprobs") or []
            start = input_token_lengths[request_idx]
            end = start + output_token_lengths[request_idx]
            selected = [
                value for value in token_logprobs[start:end]
                if value is not None
            ]
            if not selected:
                raise RuntimeError(
                    "vLLM echo logprobs did not contain output token scores."
                )
            mean_nll = -sum(float(value) for value in selected) / len(selected)
            perplexities.append(exp(mean_nll))

        if len(perplexities) != len(combined_texts):
            raise RuntimeError(
                "vLLM server returned fewer echo-logprob scores than requested."
            )
        return perplexities

    def _score(
        self,
        input_texts: List[str],
        output_texts: List[str],
        method: Literal["perplexity_score"] = "perplexity_score",
    ) -> List[float]:
        if self.cfg.scoring_mode == "prompt_logprobs":
            return self._score_with_prompt_logprobs(input_texts, output_texts)
        if self.cfg.scoring_mode == "echo_logprobs":
            return self._score_with_echo_logprobs(input_texts, output_texts)
        raise ValueError(
            f"Unsupported vLLM server scoring mode: {self.cfg.scoring_mode}"
        )
