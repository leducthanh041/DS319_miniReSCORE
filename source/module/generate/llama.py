import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gc
from typing import List, Literal, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_accelerate_available
from vllm import LLM, SamplingParams
import torch
from math import exp
from source.module.generate.utils import EOSReachedCriteria
from source.module.generate.base import BaseGenerator, BaseGeneratorConfig

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))

@dataclass
class LlamaGeneratorConfig(BaseGeneratorConfig):
    # Base Setting
    model_name: Optional[str] = 'meta-llama/Llama-3.1-8B-Instruct'
    max_total_tokens: Optional[int] = 4096
    max_model_len: Optional[int] = None
    max_new_tokens: Optional[int] = 1024
    min_new_tokens: Optional[int] = 1

    # If use Sampling
    temperature: Optional[float] = 0.
    # top_k: Optional[float] = 50
    # top_p: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    # If use Greedy decoding
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    # Tokenizer
    truncation: Optional[bool] = True
    padding: Optional[bool] = True
    # Etc
    stop: Optional[str] = field(default_factory=list)
    include_stop_str_in_output: Optional['bool'] = True
    gpu_memory_utilization: Optional[float] = 0.8
    dtype: Optional[str] = "half"
    swap_space: Optional[float] = 0
    cpu_offload_gb: Optional[float] = 0
    # vocab_size: Optional[int] = 128256 # TODO: llama vocab_size? 128256 by default
    use_vllm: Optional[bool] = True
    eos_text: Optional[str] = None
    gpu : Optional[int] = None
    device_map: Optional[str] = None
    max_memory_per_gpu: Optional[str] = None
    max_memory_map: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    

class LlamaGenerator(BaseGenerator):

    @staticmethod
    def _reraise_with_model_hint(error: Exception, model_name: str):
        message = str(error)
        if ("gated repo" in message.lower()) or ("public gated repositories" in message.lower()):
            raise OSError(
                f"Failed to access model '{model_name}'. "
                "This model is gated. If you use a fine-grained Hugging Face token, "
                "enable 'Read access to contents of all public gated repositories you can access'. "
                "A plain 'read' token also works for gated model downloads if your account already has access. "
                "Then export HF_TOKEN and rerun."
            ) from error
        raise error

    @staticmethod
    def _resolve_hf_token():
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    @staticmethod
    def _parse_max_memory_map(max_memory_map: Optional[str]):
        if not max_memory_map:
            return None

        parsed = {}
        for item in max_memory_map.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "Invalid max memory map item "
                    f"'{item}'. Expected format like '0:7GiB,1:7GiB,cpu:32GiB'."
                )
            device, memory = item.split(":", 1)
            device = device.strip()
            memory = memory.strip()
            if not device or not memory:
                raise ValueError(
                    "Invalid max memory map item "
                    f"'{item}'. Device and memory must both be set."
                )

            if device.lower() == "cpu":
                parsed["cpu"] = memory
            else:
                try:
                    parsed[int(device)] = memory
                except ValueError as error:
                    raise ValueError(
                        "Invalid max memory map device "
                        f"'{device}'. Use CUDA logical ids like 0,1,2 or 'cpu'."
                    ) from error

        if not parsed:
            raise ValueError("max_memory_map was provided but no valid entries were parsed.")

        return parsed
    
    def __init__(
        self,
        cfg: LlamaGeneratorConfig = LlamaGeneratorConfig()
    ):
        super().__init__(cfg)
        self.hf_token = self._resolve_hf_token()

        if self.cfg.gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.cfg.gpu}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.cfg.use_vllm: 
            self.vllm_max_model_len = self.cfg.max_model_len or self.cfg.max_total_tokens
            visible_gpu_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
            tensor_parallel_size = self.cfg.tensor_parallel_size
            if tensor_parallel_size is None:
                tensor_parallel_size = visible_gpu_count if self.device.type == 'cuda' else 1
            tensor_parallel_size = max(1, tensor_parallel_size)
            if self.device.type == 'cuda' and tensor_parallel_size > visible_gpu_count:
                raise ValueError(
                    f"tensor_parallel_size={tensor_parallel_size} exceeds visible CUDA "
                    f"device count={visible_gpu_count}. Check CUDA_VISIBLE_DEVICES."
                )
            self.model = LLM( 
                model=self.cfg.model_name, 
                gpu_memory_utilization=self.cfg.gpu_memory_utilization, 
                max_model_len=self.vllm_max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                dtype=self.cfg.dtype or "half",
                swap_space=self.cfg.swap_space,
                cpu_offload_gb=self.cfg.cpu_offload_gb,
                device=self.device.type,
                # enable_prefix_caching=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, 
                token=self.hf_token,
            )
            self.input_device = self.device

        else:
            self.hf_device_map = None
            model_kwargs = {
                "low_cpu_mem_usage": True,
            }
            if self.device.type == 'cuda':
                model_kwargs["torch_dtype"] = torch.float16

            if self.cfg.gpu is not None:
                self.hf_device_map = {"": self.device}
            elif self.cfg.device_map:
                if self.cfg.device_map == "auto" and not is_accelerate_available():
                    raise RuntimeError(
                        "device_map='auto' requires accelerate. Install accelerate or pass --generator_gpu."
                    )
                self.hf_device_map = self.cfg.device_map
                max_memory_map = self._parse_max_memory_map(self.cfg.max_memory_map)
                if max_memory_map:
                    model_kwargs["max_memory"] = max_memory_map
                elif self.cfg.max_memory_per_gpu and torch.cuda.is_available():
                    model_kwargs["max_memory"] = {
                        gpu_idx: self.cfg.max_memory_per_gpu
                        for gpu_idx in range(torch.cuda.device_count())
                    }
            elif self.device.type == 'cuda':
                self.hf_device_map = {"": self.device}

            if self.hf_device_map is not None:
                model_kwargs["device_map"] = self.hf_device_map
            if self.hf_token:
                model_kwargs["token"] = self.hf_token

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_name,
                    **model_kwargs
                )
            except OSError as error:
                self._reraise_with_model_hint(error, self.cfg.model_name)
            self.model.eval()  # Set the model to evaluation mode
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.model_name,
                    token=self.hf_token,
                )
            except OSError as error:
                self._reraise_with_model_hint(error, self.cfg.model_name)
            self.pad_token_initialized = False
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.input_device = self._resolve_input_device()

        self.tokenizer.padding_side = 'left'  # Set padding side to left for decoder model
            
        if self.cfg.eos_text:
            self.stopping_criteria_list = EOSReachedCriteria(
                tokenizer=self.tokenizer,
                eos_text=self.cfg.eos_text
            )
        else:
            self.stopping_criteria_list = None
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def _resolve_input_device(self):
        if self.device.type != 'cuda':
            return self.device

        if isinstance(self.hf_device_map, dict):
            for mapped_device in self.hf_device_map.values():
                if isinstance(mapped_device, int):
                    return torch.device(f'cuda:{mapped_device}')
                if isinstance(mapped_device, torch.device):
                    return mapped_device
                if isinstance(mapped_device, str) and mapped_device.startswith('cuda'):
                    return torch.device(mapped_device)

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device
    
    @torch.no_grad()
    def _generate(
        self,
        inputs = List[str],
    ):
        if self.cfg.use_vllm:
            sampling_params = SamplingParams(
                n=self.cfg.num_return_sequences,
                repetition_penalty=self.cfg.repetition_penalty,
                temperature=self.cfg.temperature,
                # top_p=self.cfg.top_p,
                # top_k=self.cfg.top_k,
                stop=[self.tokenizer.eos_token, "<|eot_id|>"] + self.cfg.stop,
                include_stop_str_in_output=self.cfg.include_stop_str_in_output,
                max_tokens=self.cfg.max_new_tokens, # Maximum number of tokens to generate per output sequence.
                min_tokens=self.cfg.min_new_tokens, # min_tokens
                truncate_prompt_tokens=max(
                    1,
                    int(self.vllm_max_model_len or self.cfg.max_total_tokens or 4096)
                    - int(self.cfg.max_new_tokens or 0),
                ),
            )
            model_outputs = self.model.generate(
                prompts=inputs, 
                sampling_params=sampling_params
            )
            generated_texts = [model_output.outputs[0].text for model_output in model_outputs]
            
        else: 
            model_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                max_length=self.cfg.max_total_tokens,
                truncation=self.cfg.truncation,
                padding=self.cfg.padding
            )
            model_inputs = {
                k:v.to(self.input_device)
                for k, v in model_inputs.items()
            }
            input_ids_length = model_inputs['input_ids'].shape[1]
            # Define the generation configuration
            do_sample = self.cfg.temperature is not None and self.cfg.temperature > 0.
            generation_args = { 
                "max_new_tokens":self.cfg.max_new_tokens, 
                "min_length": self.cfg.min_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": self.cfg.num_return_sequences,                
                "repetition_penalty": self.cfg.repetition_penalty,
                "length_penalty": self.cfg.length_penalty,
                "stopping_criteria": self.stopping_criteria_list,
                "eos_token_id": self.tokenizer.eos_token_id,  # EOS token
                "pad_token_id": self.tokenizer.pad_token_id,  # Padding token (important for batching)
            }
            if do_sample:
                generation_args["temperature"] = self.cfg.temperature
            # Generate outputs
            generated_outputs = self.model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs.get('attention_mask'),
                **generation_args
            )
            # Decode only the new tokens (exclude the input prompt)
            generated_texts = self.tokenizer.batch_decode(
                generated_outputs[:, input_ids_length:], 
                skip_special_tokens=True
            )

        outputs = generated_texts
        
        return outputs 
               
    @torch.no_grad()
    def _score(
        self, 
        input_texts: List[str],
        output_texts: List[str],
        method: Literal['perplexity_score'] = 'perplexity_score'
    ):
        if self.cfg.use_vllm:
            return self._score_with_vllm(input_texts, output_texts)

        perplexities = []
        max_total_tokens = max(2, int(self.cfg.max_total_tokens or 4096))
    
        for input_text, output_text in zip(input_texts, output_texts):
            answer_ids = self.tokenizer.encode(
                output_text, return_tensors='pt', add_special_tokens=False
            )
            if answer_ids.shape[1] == 0:
                answer_ids = torch.tensor(
                    [[self.tokenizer.eos_token_id]],
                    dtype=torch.long,
                )

            max_answer_tokens = max(1, max_total_tokens - 1)
            if answer_ids.shape[1] > max_answer_tokens:
                answer_ids = answer_ids[:, :max_answer_tokens]

            max_input_tokens = max(1, max_total_tokens - answer_ids.shape[1])
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                max_length=max_input_tokens,
                truncation=True,
            )
            input_ids = input_ids.to(self.input_device)
            answer_ids = answer_ids.to(self.input_device)
            
            total_ids = torch.cat([input_ids, answer_ids], dim=1)
            logits = self.model(total_ids).logits.squeeze(0)
            
            answer_logits = logits[input_ids.shape[1] - 1:-1, :]
            answer_labels = answer_ids.squeeze(0)[:]
            
            perplexity = torch.exp(
                F.cross_entropy(answer_logits, answer_labels) # , reduction='none'
            )
            perplexities.append(perplexity.item())  # Directly store the float
            del input_ids, answer_ids, total_ids, logits, answer_logits, answer_labels, perplexity
            
        return perplexities  # This will return a list of floats

    @staticmethod
    def _extract_vllm_logprob(logprob_entry):
        if hasattr(logprob_entry, "logprob"):
            return float(logprob_entry.logprob)
        if isinstance(logprob_entry, dict) and "logprob" in logprob_entry:
            return float(logprob_entry["logprob"])
        return float(logprob_entry)

    def _score_with_vllm(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ):
        prompt_token_ids = []
        answer_token_ids = []
        answer_offsets = []

        # vLLM needs room for at least one generated token even when we only
        # read prompt logprobs, so keep the scored prompt below max_model_len.
        max_prompt_tokens = max(
            2,
            int(getattr(self, "vllm_max_model_len", None) or self.cfg.max_total_tokens or 4096) - 1,
        )
        for input_text, output_text in zip(input_texts, output_texts):
            cur_answer_ids = self.tokenizer.encode(
                output_text,
                add_special_tokens=False,
            )
            if not cur_answer_ids:
                cur_answer_ids = [self.tokenizer.eos_token_id]

            max_answer_tokens = max(1, max_prompt_tokens - 1)
            cur_answer_ids = cur_answer_ids[:max_answer_tokens]

            max_input_tokens = max(1, max_prompt_tokens - len(cur_answer_ids))
            cur_input_ids = self.tokenizer.encode(
                input_text,
                max_length=max_input_tokens,
                truncation=True,
            )
            if not cur_input_ids:
                cur_input_ids = [self.tokenizer.eos_token_id]

            answer_offsets.append(len(cur_input_ids))
            answer_token_ids.append(cur_answer_ids)
            prompt_token_ids.append(cur_input_ids + cur_answer_ids)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=0,
            prompt_logprobs=1,
        )
        model_outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        perplexities = []
        for request_idx, model_output in enumerate(model_outputs):
            prompt_logprobs = model_output.prompt_logprobs
            cur_answer_ids = answer_token_ids[request_idx]
            answer_offset = answer_offsets[request_idx]
            token_logprobs = []

            for token_offset, token_id in enumerate(cur_answer_ids):
                position = answer_offset + token_offset
                if prompt_logprobs is None or position >= len(prompt_logprobs):
                    raise RuntimeError(
                        "vLLM did not return enough prompt_logprobs for scoring. "
                        "Try lowering generation_max_total_tokens or disable vLLM scoring."
                    )

                candidates = prompt_logprobs[position] or {}
                logprob_entry = candidates.get(token_id)
                if logprob_entry is None:
                    logprob_entry = candidates.get(str(token_id))
                if logprob_entry is None:
                    raise RuntimeError(
                        "vLLM prompt_logprobs did not include the target token. "
                        "Increase prompt_logprobs support in vLLM or disable vLLM scoring."
                    )
                token_logprobs.append(self._extract_vllm_logprob(logprob_entry))

            mean_negative_logprob = -sum(token_logprobs) / max(1, len(token_logprobs))
            perplexities.append(exp(mean_negative_logprob))

        return perplexities
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    model = LlamaGenerator(
        LlamaGeneratorConfig(
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            max_total_tokens=128,
            max_new_tokens=16,
            min_new_tokens=1,
            use_vllm=False
        )
    )
    
    inputs = [
        'Transformer-based GPTs have become extremely popular in 2024. \
            What is the most popular deep learning architecture in 2024?',
        'Diffusion-based DALL-E have become extremely popular in 2024. \
            What is the most popular deep learning architecture in 2024?',
        'Does Roh Tae-woo died earlier then Jun Duhwan?',
        'President often live long\nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Roh Tae-woo died in 21/10/26\nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Jun Duhwan died in 21/11/23 \nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Roh Tae-woo died in 21/10/26\nJun Duhwan died in 21/11/23\nDoes Roh Tae-woo died earlier then Jun Duhwan?'
    ]
    
    sample_texts = [
        model.tokenizer.apply_chat_template([{'role': 'user', 'content': i}], tokenize=False, add_generation_prompt=True).replace('<|begin_of_text|>', '')
        for i in inputs
    ]
    
    sample_forced_outputs = [
        'So the answer is: Transformer',
        'So the answer is: Transformer',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
    ]
    
    scores = model.score(
        input_texts=sample_texts,
        output_texts=sample_forced_outputs
    )
    
    print("\nPEREPLEXITY RESULTS\n")

    for example in zip(sample_texts, sample_forced_outputs, scores):
        print(example)
