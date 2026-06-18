import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import copy
import json
import multiprocessing as mp
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, fields
from datetime import datetime
from math import exp
from typing import Any, Dict, List, Literal, Optional

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from source.utility.data_utils import clean_and_create_dir, load_data_from_jsonl
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState
from source.pipeline.step.retrieval import RetrievalStep
from source.pipeline.step.training import TrainStep
from source.pipeline.step.generation import (
    AnswerGenerateOutputParser,
    AnswerGeneratePromptGenerator,
    GenerationStep,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.end import EndStep
from source.module.generate.base import BaseGenerator, BaseGeneratorConfig
from source.module.generate.llama import LlamaGenerator, LlamaGeneratorConfig
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.module.index.index import Indexer, IndexerConfig


def collate_question_states(batch: List[QuestionState]) -> List[QuestionState]:
    return batch


class QuestionStateDataset(Dataset):
    def __init__(self, start_states):
        self.start_states = start_states

    def __len__(self):
        return len(self.start_states)

    def __getitem__(self, idx):
        return self.start_states[idx]


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


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
    include_stop_str_in_output: bool = True
    request_timeout: float = 600.0
    scoring_mode: str = "prompt_logprobs"
    prompt_logprobs: int = 20
    missing_logprob_fallback: float = -20.0


class VLLMServerGenerator(BaseGenerator):
    """Generator client for a persistent vLLM OpenAI-compatible server."""

    def __init__(self, cfg: VLLMServerGeneratorConfig):
        super().__init__(cfg)
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.served_model_name or cfg.model_name
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            token=self.hf_token,
        )
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
            payload_summary = self._summarize_payload(payload)
            print(
                "[vllm-server-error] "
                f"url={url} status={error.code} reason={error.reason} "
                f"payload={payload_summary}",
                file=sys.stderr,
            )
            if body:
                print(
                    "[vllm-server-error-body] "
                    f"{body[:4000]}",
                    file=sys.stderr,
                )
            raise RuntimeError(f"vLLM server request failed: {error.code} {body}") from error
        except urllib.error.URLError as error:
            print(
                "[vllm-server-error] "
                f"url={url} connection_error={error}",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"Cannot connect to vLLM server at {self.base_url}. "
                "Start it first with script/preload_vllm_server.py."
            ) from error

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            prompt_count = len(prompt)
            prompt_chars = sum(len(str(item)) for item in prompt)
            prompt_tokens = sum(
                len(self.tokenizer.encode(str(item), add_special_tokens=False))
                for item in prompt[: min(4, len(prompt))]
            )
            prompt_tokens_estimate = (
                int(prompt_tokens * prompt_count / min(4, prompt_count))
                if prompt_count > 0
                else 0
            )
        elif prompt is None:
            prompt_count = 0
            prompt_chars = 0
            prompt_tokens_estimate = 0
        else:
            prompt_count = 1
            prompt_chars = len(str(prompt))
            prompt_tokens_estimate = len(
                self.tokenizer.encode(str(prompt), add_special_tokens=False)
            )

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
        try:
            self._post_json("/completions", payload)
        except RuntimeError as error:
            raise RuntimeError(
                "vLLM server is reachable but cannot complete a minimal request. "
                "Restart the preloaded server with a smaller max_model_len, lower "
                "gpu_memory_utilization, or --enforce_eager. "
                f"Original error: {error}"
            ) from error

    @staticmethod
    def _choice_texts_by_index(response: Dict[str, Any], expected_count: int) -> List[str]:
        outputs = [""] * expected_count
        for fallback_idx, choice in enumerate(response.get("choices", [])):
            index = int(choice.get("index", fallback_idx))
            if 0 <= index < expected_count:
                outputs[index] = choice.get("text", "")
        return outputs

    @staticmethod
    def _extract_logprob(logprob_entry: Any) -> float:
        if isinstance(logprob_entry, dict):
            if "logprob" in logprob_entry:
                return float(logprob_entry["logprob"])
            if "log_prob" in logprob_entry:
                return float(logprob_entry["log_prob"])
        return float(logprob_entry)

    def _generate(self, inputs: List[str]) -> List[str]:
        stop = [self.tokenizer.eos_token, "<|eot_id|>"] + list(self.cfg.stop)
        stop = [item for item in stop if item]
        max_prompt_tokens = max(
            1,
            int(self.cfg.max_total_tokens) - int(self.cfg.max_new_tokens),
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
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "stop": stop,
            "stream": False,
        }
        response = self._post_json("/completions", payload)
        return self._choice_texts_by_index(response, len(inputs))

    def _score_with_prompt_logprobs(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> List[float]:
        combined_texts = []
        answer_token_ids = []
        answer_offsets = []
        max_prompt_tokens = max(2, int(self.cfg.score_max_tokens) - 1)

        for input_text, output_text in zip(input_texts, output_texts):
            cur_answer_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
            if not cur_answer_ids:
                cur_answer_ids = [self.tokenizer.eos_token_id]

            cur_answer_ids = cur_answer_ids[: max(1, max_prompt_tokens - 1)]
            max_input_tokens = max(1, max_prompt_tokens - len(cur_answer_ids))
            cur_input_ids = self.tokenizer.encode(
                input_text,
                max_length=max_input_tokens,
                truncation=True,
                add_special_tokens=False,
            )
            if not cur_input_ids:
                cur_input_ids = [self.tokenizer.eos_token_id]

            answer_offsets.append(len(cur_input_ids))
            answer_token_ids.append(cur_answer_ids)
            combined_texts.append(
                self.tokenizer.decode(
                    cur_input_ids + cur_answer_ids,
                    skip_special_tokens=False,
                )
            )

        payload = {
            "model": self.model,
            "prompt": combined_texts,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": self.cfg.prompt_logprobs,
            "stream": False,
        }
        response = self._post_json("/completions", payload)

        choices = sorted(response.get("choices", []), key=lambda choice: int(choice.get("index", 0)))
        if len(choices) < len(combined_texts):
            raise RuntimeError("vLLM server returned fewer scoring choices than requested.")

        perplexities = []
        for request_idx, choice in enumerate(choices[: len(combined_texts)]):
            prompt_logprobs = choice.get("prompt_logprobs")
            if not prompt_logprobs:
                raise RuntimeError("vLLM server response does not contain prompt_logprobs.")

            token_logprobs = []
            for token_offset, token_id in enumerate(answer_token_ids[request_idx]):
                position = answer_offsets[request_idx] + token_offset
                if position >= len(prompt_logprobs):
                    if not self._warned_incomplete_prompt_logprobs:
                        print(
                            "[vllm-server-warning] vLLM server returned incomplete "
                            "prompt_logprobs; using fallback logprob for missing "
                            "answer-token positions. "
                            f"expected_position={position} "
                            f"returned_positions={len(prompt_logprobs)} "
                            f"fallback={self.cfg.missing_logprob_fallback}",
                            file=sys.stderr,
                        )
                        self._warned_incomplete_prompt_logprobs = True
                    token_logprobs.append(float(self.cfg.missing_logprob_fallback))
                    continue
                candidates = prompt_logprobs[position] or {}
                logprob_entry = candidates.get(str(token_id))
                if logprob_entry is None:
                    logprob_entry = candidates.get(token_id)
                if logprob_entry is None:
                    if not self._warned_score_fallback:
                        decoded_token = self.tokenizer.decode([token_id])
                        print(
                            "[vllm-server-warning] Target token missing from "
                            "prompt_logprobs; using fallback logprob. "
                            f"token_id={token_id} decoded={decoded_token!r} "
                            f"prompt_logprobs={self.cfg.prompt_logprobs} "
                            f"fallback={self.cfg.missing_logprob_fallback}",
                            file=sys.stderr,
                        )
                        self._warned_score_fallback = True
                    token_logprobs.append(float(self.cfg.missing_logprob_fallback))
                else:
                    token_logprobs.append(self._extract_logprob(logprob_entry))

            mean_negative_logprob = -sum(token_logprobs) / max(1, len(token_logprobs))
            perplexities.append(exp(mean_negative_logprob))

        return perplexities

    def _score_with_echo_logprobs(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> List[float]:
        combined_texts = []
        input_token_lengths = []
        answer_token_lengths = []
        max_prompt_tokens = max(2, int(self.cfg.score_max_tokens) - 1)

        for input_text, output_text in zip(input_texts, output_texts):
            answer_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
            if not answer_ids:
                answer_ids = [self.tokenizer.eos_token_id]
            answer_ids = answer_ids[: max(1, max_prompt_tokens - 1)]

            max_input_tokens = max(1, max_prompt_tokens - len(answer_ids))
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=max_input_tokens,
                truncation=True,
                add_special_tokens=False,
            )
            if not input_ids:
                input_ids = [self.tokenizer.eos_token_id]

            combined_texts.append(
                self.tokenizer.decode(
                    input_ids + answer_ids,
                    skip_special_tokens=False,
                )
            )
            input_token_lengths.append(len(input_ids))
            answer_token_lengths.append(len(answer_ids))
        payload = {
            "model": self.model,
            "prompt": combined_texts,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
            "stream": False,
        }
        response = self._post_json("/completions", payload)
        choices = sorted(response.get("choices", []), key=lambda choice: int(choice.get("index", 0)))

        perplexities = []
        for request_idx, choice in enumerate(choices[: len(combined_texts)]):
            logprobs = choice.get("logprobs") or {}
            token_logprobs = logprobs.get("token_logprobs") or []
            text_offsets = logprobs.get("text_offset") or []
            answer_char_start = len(input_texts[request_idx])

            if text_offsets and len(text_offsets) == len(token_logprobs):
                selected = [
                    value for value, offset in zip(token_logprobs, text_offsets)
                    if value is not None and offset >= answer_char_start
                ]
            else:
                start = input_token_lengths[request_idx]
                end = start + answer_token_lengths[request_idx]
                selected = [
                    value for value in token_logprobs[start:end]
                    if value is not None
                ]
            if not selected:
                raise RuntimeError("vLLM echo logprobs did not contain answer token scores.")
            mean_negative_logprob = -sum(float(value) for value in selected) / len(selected)
            perplexities.append(exp(mean_negative_logprob))

        if len(perplexities) != len(combined_texts):
            raise RuntimeError("vLLM server returned fewer echo-logprob scores than requested.")
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
        raise ValueError(f"Unsupported vLLM server scoring mode: {self.cfg.scoring_mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReSCORE retriever")

    parser.add_argument("--method", type=str, default="rescore", help="Method name")
    parser.add_argument("--running_name", type=str, default=None, help="Name for the run")
    parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "2wikimultihopqa", "musique"],
        default="musique",
        help="Dataset",
    )
    parser.add_argument(
        "--dataset_split",
        choices=["train", "dev", "test"],
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--pipeline_type",
        choices=["single_retrieval", "multi_retrieval", "no_retrieval"],
        default="multi_retrieval",
        help="Pipeline type",
    )

    parser.add_argument("--prompt_set", type=int, default=1, help="Prompt set")
    parser.add_argument(
        "--prompt_document_from",
        choices=["last_only", "full"],
        default="last_only",
        help="Document selection for prompts",
    )
    parser.add_argument("--prompt_max_para_count", type=int, default=15, help="Max paragraph count")
    parser.add_argument("--prompt_max_para_words", type=int, default=350, help="Max words per paragraph")

    parser.add_argument(
        "--generation_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Generation model",
    )
    parser.add_argument(
        "--generation_backend",
        choices=["vllm_server", "local"],
        default="vllm_server",
        help="Use a persistent vLLM server client or load the generator locally in this process.",
    )
    parser.add_argument(
        "--vllm_server_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM server base URL used when generation_backend=vllm_server.",
    )
    parser.add_argument(
        "--vllm_server_model",
        type=str,
        default=None,
        help="Served model name exposed by the vLLM server; defaults to generation_model_name.",
    )
    parser.add_argument(
        "--vllm_server_timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout in seconds for the vLLM server client.",
    )
    parser.add_argument(
        "--vllm_server_score_max_tokens",
        type=int,
        default=1024,
        help="Max tokens used by vLLM-server scoring requests. Lower this to avoid logprobs OOM.",
    )
    parser.add_argument(
        "--vllm_server_scoring_mode",
        choices=["prompt_logprobs", "echo_logprobs"],
        default="prompt_logprobs",
        help="Scoring API mode for the external vLLM server.",
    )
    parser.add_argument(
        "--vllm_server_prompt_logprobs",
        type=int,
        default=20,
        help="Top-k prompt logprobs requested from vLLM server scoring.",
    )
    parser.add_argument(
        "--vllm_server_missing_logprob_fallback",
        type=float,
        default=-20.0,
        help="Fallback logprob when a target token is outside returned prompt_logprobs top-k.",
    )
    parser.add_argument(
        "--generation_max_batch_size",
        type=int,
        default=1,
        help="Batch size for generation/scoring",
    )
    parser.add_argument(
        "--generation_max_total_tokens",
        type=int,
        default=4096,
        help="Max total tokens for generation/scoring",
    )
    parser.add_argument(
        "--generation_max_model_len",
        type=int,
        default=4096,
        help="vLLM max_model_len/KV-cache length when --generation_use_vllm is enabled. Use 2048/1024 on 11GB GPUs.",
    )
    parser.add_argument("--generation_max_new_tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--generation_min_new_tokens", type=int, default=1, help="Min new tokens")
    parser.add_argument(
        "--generation_device_map",
        type=str,
        default="auto",
        help="HF device_map for generator; use 'auto' to shard across visible GPUs",
    )
    parser.add_argument(
        "--generation_max_memory_per_gpu",
        type=str,
        default="10GiB",
        help="Max memory per visible GPU for generator when device_map=auto",
    )
    parser.add_argument(
        "--generation_max_memory_map",
        type=str,
        default=None,
        help="Per-device generator max memory map, for example '0:7GiB,1:7GiB,2:7GiB' to reserve cuda:3",
    )
    parser.add_argument(
        "--generation_use_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM for LM generation/scoring. This switches scoring to vLLM prompt_logprobs.",
    )
    parser.add_argument(
        "--generation_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory utilization when --generation_use_vllm is enabled",
    )
    parser.add_argument(
        "--generation_swap_space",
        type=float,
        default=0,
        help="vLLM CPU swap space in GiB per GPU when --generation_use_vllm is enabled",
    )
    parser.add_argument(
        "--generation_cpu_offload_gb",
        type=float,
        default=0,
        help="vLLM CPU offload size in GiB when --generation_use_vllm is enabled",
    )
    parser.add_argument(
        "--generation_dtype",
        type=str,
        default="half",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="vLLM model dtype when --generation_use_vllm is enabled. Use half/float16 for RTX 2080 Ti.",
    )
    parser.add_argument(
        "--generation_tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size when --generation_use_vllm is enabled; defaults to all visible GPUs",
    )
    parser.add_argument("--generator_gpu", type=int, default=None, help="Single GPU id for generator")

    parser.add_argument(
        "--retrieval_query_type",
        choices=["last_only", "full"],
        default="full",
        help="Query type for retrieval",
    )
    parser.add_argument("--retrieval_count", type=int, choices=[2, 4, 6, 8], default=8, help="Retrieval count")
    parser.add_argument("--retrieval_buffer_size", type=int, default=32, help="Retriever buffer size")
    parser.add_argument("--retrieval_no_duplicates", action="store_true", help="Disable duplicates")
    parser.add_argument(
        "--retrieval_no_reasoning_sentences",
        action="store_true",
        help="Drop reasoning sentences in retrieval query",
    )
    parser.add_argument("--retrieval_no_wh_words", action="store_true", help="Drop WH words in retrieval query")

    parser.add_argument(
        "--retrieval_query_model_name_or_path",
        type=str,
        default="facebook/contriever-msmarco",
        help="Retriever query encoder",
    )
    parser.add_argument(
        "--retrieval_passage_model_name_or_path",
        type=str,
        default=None,
        help="Retriever passage encoder",
    )
    parser.add_argument("--retrieval_batch_size", type=int, default=32, help="Retriever embedding batch size")
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Override retrieval DB directory containing docstore.db, index.faiss, and faiss_id_to_docstore_id.pkl",
    )
    parser.add_argument(
        "--retrieval_training_strategy",
        choices=["query_only", "both"],
        default="query_only",
        help="Retriever training strategy",
    )
    parser.add_argument(
        "--retrieval_use_fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 for the frozen passage encoder",
    )
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Retriever device, for example cuda:3 or cpu",
    )

    parser.add_argument("--max_num_thought", type=int, default=6, help="Max number of thoughts")
    parser.add_argument("--answer_regex", type=str, default=".* Answer: <.*>\\.?", help="Answer regex")
    parser.add_argument("--match_all_on_failure", action="store_true", help="Regex fallback")
    parser.add_argument("--demo", action="store_true", help="Use demo subset")

    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--temperature_r", type=float, default=0.1, help="Retriever temperature")
    parser.add_argument("--temperature_lm", type=float, default=1.0, help="LM temperature")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")

    parser.add_argument("--num_workers", type=int, default=40, help="DataLoader workers")
    parser.add_argument("--validation_freq", type=int, default=10, help="Run validation every N steps")
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Validation batch size; defaults to training batch size",
    )
    parser.add_argument(
        "--validation_max_batches",
        type=int,
        default=0,
        help="Limit validation to N batches; <=0 means full dev set",
    )
    parser.add_argument("--save_freq", type=int, default=10, help="Save retriever every N steps")
    parser.add_argument("--runtime_log_root", type=str, default="./logs/train", help="Runtime log root directory")
    parser.add_argument(
        "--prediction_root",
        type=str,
        default=None,
        help="Override root directory for predictions/checkpoints, for example /docker/data/$USER/ReSCORE/predictions.",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop when validation loss stops improving",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop after this many validation checks without improvement",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss improvement to reset early-stopping patience",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Stop this run after N training batches; <=0 means no explicit step limit.",
    )

    return parser.parse_args()


def build_pipeline_config(args):
    cfg_field_names = {field.name for field in fields(PipelineConfig)}
    cfg_kwargs = {
        key: value for key, value in vars(args).items()
        if key in cfg_field_names
    }
    if args.database_path:
        cfg_kwargs["database_path_override"] = args.database_path
    if args.prediction_root:
        cfg_kwargs["prediction_root_override"] = args.prediction_root
    cfg_kwargs["train"] = True
    if not cfg_kwargs.get("running_name"):
        cfg_kwargs["running_name"] = f"train_{cfg_kwargs['dataset']}"
    return PipelineConfig(**cfg_kwargs)


def get_pipeline(cfg, contexts, generator, retriever, indexer):
    if cfg.pipeline_type == "no_retrieval":
        raise NotImplementedError("no_retrieval is not implemented for training.")

    if cfg.pipeline_type == "single_retrieval":
        raise NotImplementedError("single_retrieval is not implemented for training.")

    if cfg.pipeline_type != "multi_retrieval":
        raise ValueError(f"Unsupported pipeline_type: {cfg.pipeline_type}")

    return [
        RetrievalStep(
            cfg=cfg,
            retriever=retriever,
            indexer=indexer,
        ),
        TrainStep(
            cfg,
            generator=generator,
            retriever=retriever,
            indexer=indexer,
        ),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=AnswerGeneratePromptGenerator(cfg),
            output_parser=AnswerGenerateOutputParser(cfg),
        ),
        EndStep(
            cfg=cfg,
        ),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=ThoughtGeneratePromptGenerator(cfg),
            output_parser=ThoughtGenerateOutputParser(cfg),
        ),
    ]


def build_runtime_log_path(runtime_log_root, dataset, running_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = running_name or f"train_{dataset}"
    return os.path.join(runtime_log_root, dataset, f"{run_name}__{timestamp}", "train.log")


def resolve_retriever_device(requested_device):
    if requested_device:
        return requested_device
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() > 1:
        return f"cuda:{torch.cuda.device_count() - 1}"
    return "cuda:0"


def ensure_runtime_paths(cfg):
    required_files = [
        cfg.data_file_path,
        cfg.qa_gen_input_prompt_file_path,
        cfg.qa_gen_output_prompt_file_path,
    ]
    for path in required_files:
        if not os.path.exists(path):
            if path.startswith(f"./data/processed_data/{cfg.dataset}/"):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Generate processed QA data first with:\n"
                    "bash script/download/multihop_processed_data.sh\n"
                    f"or rerun:\n"
                    "python ./preprocess/process_<dataset>.py\n"
                    "python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name <dataset> --set_name dev\n"
                    "python ./preprocess/subsample_dataset_and_remap_paras.py --dataset_name <dataset> --set_name test"
                )
            raise FileNotFoundError(f"Required file not found: {path}")

    if not os.path.isdir(cfg.database_path):
        raise FileNotFoundError(f"Database directory not found: {cfg.database_path}")

    database_artifacts = [
        os.path.join(cfg.database_path, "docstore.db"),
        os.path.join(cfg.database_path, "index.faiss"),
        os.path.join(cfg.database_path, "faiss_id_to_docstore_id.pkl"),
    ]
    missing_artifacts = [path for path in database_artifacts if not os.path.exists(path)]
    if missing_artifacts:
        raise FileNotFoundError(
            "Missing index artifacts:\n" + "\n".join(missing_artifacts)
        )


def describe_cuda_environment():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    print(f"Visible CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            total_gib = props.total_memory / (1024 ** 3)
            print(
                f"Logical GPU {gpu_idx}: {props.name}, total_memory={total_gib:.2f} GiB"
            )


def log_hot_storage_paths(runtime_log_path, cfg):
    tracked_paths = {
        "runtime_log_dir": os.path.dirname(runtime_log_path),
        "prediction_dir": cfg.prediction_file_dir,
        "database_dir": cfg.database_path,
    }
    print("Hot storage layout:")
    for label, path in tracked_paths.items():
        real_path = os.path.realpath(path)
        location = "/docker" if real_path.startswith("/docker/") or real_path == "/docker" else "non-/docker"
        print(f"  {label}: path={path} realpath={real_path} [{location}]")

    runtime_log_real_path = os.path.realpath(tracked_paths["runtime_log_dir"])
    if not (runtime_log_real_path.startswith("/docker/") or runtime_log_real_path == "/docker"):
        print(
            "[warning] Runtime logs are not on /docker. "
            "To avoid NFS-induced D-state hangs, run: bash script/setup_local_hot_data.sh"
        )


def reset_controller_state(controller):
    controller.state_tree.clear()
    controller.running_state_ids.clear()
    controller.end_state_ids.clear()


def safe_cuda_empty_cache(context):
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except RuntimeError as error:
        print(
            f"[cuda-error] CUDA cache cleanup failed at {context}. "
            "This usually means an earlier CUDA kernel failed asynchronously."
        )
        print(f"[cuda-error] {error}")
        print(
            "[cuda-error] Stop this run and resume from the latest/best checkpoint. "
            "For exact fault location, rerun a short debug pass with CUDA_LAUNCH_BLOCKING=1."
        )
        raise


def build_start_states(inputs, id_to_ground_truths):
    return [
        QuestionState(
            question_id=question_id,
            question=question_text,
            answer=id_to_ground_truths[question_id][0],
        )
        for question_id, question_text in inputs.items()
    ]


def build_dataloader(start_states, batch_size, num_workers):
    return DataLoader(
        QuestionStateDataset(start_states),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_question_states,
    )


def validate(
    cfg,
    controller,
    epoch,
    num_steps,
    demo,
    num_workers,
    validation_batch_size=None,
    validation_max_batches=0,
):
    total_loss = 0.0
    total_batches = 0

    retriever = controller.pipeline[0].retriever
    query_model_was_training = retriever.query_model.training
    retriever.query_model.eval()

    validation_cfg = copy.deepcopy(cfg)
    validation_cfg.dataset_split = "dev"

    dev_inputs, dev_id_to_ground_truths, _ = load_data_from_jsonl(
        file_path=validation_cfg.data_file_path,
        ground_truth_file_path=validation_cfg.id_to_ground_truths_file_path,
        return_contexts=True,
        is_demo=demo,
    )
    dev_start_states = build_start_states(dev_inputs, dev_id_to_ground_truths)
    dev_batch_size = validation_batch_size or cfg.batch_size
    dev_dataloader = build_dataloader(dev_start_states, dev_batch_size, num_workers)

    with torch.no_grad():
        for batch in dev_dataloader:
            batch_loss = controller.train(batch)
            total_loss += batch_loss.item()
            total_batches += 1
            reset_controller_state(controller)
            if validation_max_batches > 0 and total_batches >= validation_max_batches:
                break

    avg_loss = total_loss / max(total_batches, 1)
    limit_msg = (
        f" limited_to={validation_max_batches}"
        if validation_max_batches > 0
        else ""
    )
    print(
        f"[validation] epoch={epoch} step={num_steps} "
        f"avg_loss={avg_loss:.6f} batches={total_batches}{limit_msg}"
    )

    retriever.query_model.train(query_model_was_training)
    safe_cuda_empty_cache(f"validation epoch={epoch} step={num_steps}")

    return avg_loss


def train(cfg, generator, retriever, indexer, optimizer, scheduler, args):
    cfg.dataset_split = "train"

    clean_and_create_dir(cfg.prediction_file_dir)
    cfg.save()

    inputs, id_to_ground_truths, contexts = load_data_from_jsonl(
        file_path=cfg.data_file_path,
        ground_truth_file_path=cfg.ground_truth_file_path,
        return_contexts=True,
        is_demo=cfg.demo,
    )

    controller = PipelineController(
        pipeline=get_pipeline(cfg, contexts, generator, retriever, indexer),
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )

    start_states = build_start_states(inputs, id_to_ground_truths)
    dataloader = build_dataloader(start_states, cfg.batch_size, args.num_workers)

    optimizer.zero_grad(set_to_none=True)
    num_accumulations = 0
    optimizer_steps = 0

    print(f"Training examples: {len(start_states)}")
    print(f"Prediction directory: {cfg.prediction_file_dir}")
    print(f"Pipeline JSON log: {cfg.logging_file_path}")
    print(
        "Validation config: "
        f"freq={args.validation_freq}, "
        f"batch_size={args.validation_batch_size or cfg.batch_size}, "
        f"max_batches={args.validation_max_batches if args.validation_max_batches > 0 else 'full'}, "
        f"early_stopping={args.early_stopping}, "
        f"patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta}"
    )

    best_val_loss = float("inf")
    bad_validation_count = 0
    should_stop = False
    global_steps_this_run = 0
    best_path = os.path.join(cfg.prediction_file_dir, "best_validation")
    best_checkpoint_saved = False
    if args.early_stopping and args.validation_freq <= 0:
        print("[early-stopping] disabled because validation_freq <= 0")
    if args.max_train_steps > 0:
        print(f"[max-train-steps] enabled max_train_steps={args.max_train_steps}")

    for epoch in range(cfg.n_epochs):
        num_steps = 0
        print(f"[epoch-start] epoch={epoch}")

        for batch in dataloader:
            num_steps += 1
            global_steps_this_run += 1
            batch_loss = controller.train(batch)
            loss_value = batch_loss.item()
            batch_loss.backward()
            num_accumulations += 1

            print(
                f"[train] epoch={epoch} step={num_steps} "
                f"loss={loss_value:.6f} accumulation={num_accumulations}/{cfg.gradient_accumulation_steps}"
            )

            if cfg.wandb_key:
                log = {
                    "training_loss": loss_value,
                    "epoch": epoch,
                    "step": num_steps,
                }
                if scheduler is not None:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)

            if num_accumulations >= cfg.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                num_accumulations = 0
                optimizer_steps += 1

                if scheduler is not None and optimizer_steps % 100 == 0:
                    scheduler.step()
                    print(
                        f"[scheduler] optimizer_step={optimizer_steps} lr={scheduler.get_last_lr()[0]:.8f}"
                    )

            reset_controller_state(controller)

            if args.validation_freq > 0 and num_steps % args.validation_freq == 0:
                val_loss = validate(
                    cfg,
                    controller,
                    epoch,
                    num_steps,
                    cfg.demo,
                    args.num_workers,
                    validation_batch_size=args.validation_batch_size,
                    validation_max_batches=args.validation_max_batches,
                )

                if cfg.wandb_key:
                    wandb.log(
                        {
                            "validation_loss": val_loss,
                            "epoch": epoch,
                            "step": num_steps,
                        }
                    )

                if args.early_stopping:
                    improved = val_loss < best_val_loss - args.early_stopping_min_delta
                    if improved:
                        best_val_loss = val_loss
                        bad_validation_count = 0
                        clean_and_create_dir(best_path)
                        retriever.query_model.save_pretrained(best_path)
                        retriever.query_tokenizer.save_pretrained(best_path)
                        best_checkpoint_saved = True
                        print(
                            f"[early-stopping] improvement val_loss={val_loss:.6f}; "
                            f"saved best retriever to {best_path}"
                        )
                    else:
                        bad_validation_count += 1
                        print(
                            f"[early-stopping] no improvement "
                            f"({bad_validation_count}/{args.early_stopping_patience}); "
                            f"best_val_loss={best_val_loss:.6f}"
                        )
                        if bad_validation_count >= args.early_stopping_patience:
                            print(
                                f"[early-stopping] stopping at epoch={epoch} step={num_steps} "
                                f"after {bad_validation_count} validation checks without improvement"
                            )
                            should_stop = True
                            break

            if num_steps % 20 == 0:
                safe_cuda_empty_cache(f"train epoch={epoch} step={num_steps}")

            if args.max_train_steps > 0 and global_steps_this_run >= args.max_train_steps:
                print(
                    f"[max-train-steps] stopping at epoch={epoch} "
                    f"step={num_steps} run_step={global_steps_this_run}"
                )
                should_stop = True
                break

        if should_stop:
            if num_accumulations > 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                num_accumulations = 0
                print(f"[optimizer] flushed trailing gradients before stop at epoch={epoch}")
            else:
                optimizer.zero_grad(set_to_none=True)
            print(f"[epoch-end] epoch={epoch} stopped=True optimizer_steps={optimizer_steps}")
            break

        if num_accumulations > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            num_accumulations = 0
            print(f"[optimizer] flushed trailing gradients at epoch={epoch}")

        print(f"[epoch-end] epoch={epoch} optimizer_steps={optimizer_steps}")

    if not best_checkpoint_saved:
        clean_and_create_dir(best_path)
        retriever.query_model.save_pretrained(best_path)
        retriever.query_tokenizer.save_pretrained(best_path)
        print(f"[done] no validation improvement recorded; saved final retriever to {best_path}")
    else:
        print(f"[done] best retriever remains at {best_path}")


def main():
    args = parse_args()
    cfg = build_pipeline_config(args)

    runtime_log_path = build_runtime_log_path(args.runtime_log_root, cfg.dataset, cfg.running_name)
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)

    with open(runtime_log_path, "a", encoding="utf-8", buffering=1) as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"Runtime log: {runtime_log_path}")
            print(f"Run name: {cfg.running_name}")
            print(f"Dataset: {cfg.dataset}")
            print(f"Dataset split: {cfg.dataset_split}")
            print(f"Data file: {cfg.data_file_path}")
            print(f"Database path: {cfg.database_path}")
            print(f"Prompt set: {cfg.prompt_set}")
            log_hot_storage_paths(runtime_log_path, cfg)
            ensure_runtime_paths(cfg)
            print(
                "VRAM-safe defaults: "
                f"batch_size={cfg.batch_size}, "
                f"generation_max_batch_size={cfg.generation_max_batch_size}, "
                f"generation_max_total_tokens={cfg.generation_max_total_tokens}, "
                f"retrieval_batch_size={cfg.retrieval_batch_size}, "
                f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}"
            )
            describe_cuda_environment()

            generation_device_map = None if args.generator_gpu is not None else args.generation_device_map
            retriever_device = resolve_retriever_device(args.retriever_device)

            if args.generation_backend == "vllm_server":
                print(
                    "Generator config: "
                    f"backend={args.generation_backend}, "
                    f"model={cfg.generation_model_name}, "
                    f"url={args.vllm_server_url}, "
                    f"served_model={args.vllm_server_model or cfg.generation_model_name}, "
                    f"score_max_tokens={args.vllm_server_score_max_tokens}, "
                    f"scoring_mode={args.vllm_server_scoring_mode}, "
                    f"prompt_logprobs={args.vllm_server_prompt_logprobs}, "
                    f"request_timeout={args.vllm_server_timeout}, "
                    "local_vllm_load=False"
                )
            else:
                print(
                    "Generator config: "
                    f"backend={args.generation_backend}, "
                    f"model={cfg.generation_model_name}, "
                    f"use_vllm={args.generation_use_vllm}, "
                    f"device_map={generation_device_map}, "
                    f"generator_gpu={args.generator_gpu}, "
                    f"max_model_len={args.generation_max_model_len or cfg.generation_max_total_tokens}, "
                    f"max_memory_per_gpu={args.generation_max_memory_per_gpu}, "
                    f"max_memory_map={args.generation_max_memory_map}, "
                    f"gpu_memory_utilization={args.generation_gpu_memory_utilization}, "
                    f"swap_space={args.generation_swap_space}, "
                    f"cpu_offload_gb={args.generation_cpu_offload_gb}, "
                    f"dtype={args.generation_dtype}, "
                    f"tensor_parallel_size={args.generation_tensor_parallel_size or 'all_visible'}"
                )
            print(
                "Hugging Face auth: "
                f"HF_HOME={os.environ.get('HF_HOME', '(default)')}, "
                f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'unset'}, "
                f"HUGGINGFACE_HUB_TOKEN={'set' if os.environ.get('HUGGINGFACE_HUB_TOKEN') else 'unset'}"
            )
            print(
                "Retriever config: "
                f"model={cfg.retrieval_query_model_name_or_path}, "
                f"device={retriever_device}, "
                f"fp16={cfg.retrieval_use_fp16}"
            )

            if args.generation_backend == "vllm_server":
                generator = VLLMServerGenerator(
                    VLLMServerGeneratorConfig(
                        model_name=cfg.generation_model_name,
                        served_model_name=args.vllm_server_model,
                        base_url=args.vllm_server_url,
                        batch_size=cfg.generation_max_batch_size,
                        max_total_tokens=cfg.generation_max_total_tokens,
                        score_max_tokens=args.vllm_server_score_max_tokens,
                        max_new_tokens=cfg.generation_max_new_tokens,
                        min_new_tokens=cfg.generation_min_new_tokens,
                        request_timeout=args.vllm_server_timeout,
                        scoring_mode=args.vllm_server_scoring_mode,
                        prompt_logprobs=args.vllm_server_prompt_logprobs,
                        missing_logprob_fallback=args.vllm_server_missing_logprob_fallback,
                    )
                )
            else:
                generator = LlamaGenerator(
                    LlamaGeneratorConfig(
                        model_name=cfg.generation_model_name,
                        batch_size=cfg.generation_max_batch_size,
                        max_total_tokens=cfg.generation_max_total_tokens,
                        max_model_len=args.generation_max_model_len,
                        max_new_tokens=cfg.generation_max_new_tokens,
                        min_new_tokens=cfg.generation_min_new_tokens,
                        use_vllm=args.generation_use_vllm,
                        gpu_memory_utilization=args.generation_gpu_memory_utilization,
                        swap_space=args.generation_swap_space,
                        cpu_offload_gb=args.generation_cpu_offload_gb,
                        dtype=args.generation_dtype,
                        tensor_parallel_size=args.generation_tensor_parallel_size,
                        gpu=args.generator_gpu,
                        device_map=generation_device_map,
                        max_memory_per_gpu=args.generation_max_memory_per_gpu,
                        max_memory_map=args.generation_max_memory_map,
                    )
                )
            retriever = DenseRetriever(
                DenseRetrieverConfig(
                    query_model_name_or_path=cfg.retrieval_query_model_name_or_path,
                    passage_model_name_or_path=cfg.retrieval_passage_model_name_or_path,
                    batch_size=cfg.retrieval_batch_size,
                    training_strategy=cfg.retrieval_training_strategy,
                    use_fp16=cfg.retrieval_use_fp16,
                    device=retriever_device,
                )
            )
            indexer = Indexer.load_local(
                IndexerConfig(
                    embedding_sz=768,
                    database_path=cfg.database_path,
                )
            )

            optimizer = AdamW(
                retriever.query_model.parameters(),
                lr=cfg.lr,
            )
            scheduler = ExponentialLR(optimizer, gamma=0.9)

            train(cfg, generator, retriever, indexer, optimizer, scheduler, args)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
