"""
inference_tta.py
================
TTA inference entry point cho ReSCORE-TTA.

Khác inference.py:
  1. Load thêm CrossEncoderWrapper cho CE pseudo-labels
  2. Inject LoRA vào query encoder (nếu tta_level in ['l2', 'both'])
  3. Dùng TTARetrievalStep thay RetrievalStep
  4. Force batch_size=1 nếu Level 2 active (per-instance LoRA)
  5. Thêm TTA-specific CLI args

Cách chạy (ví dụ):
    python -m source.run.inference_tta \\
        --method iqatr_tta \\
        --dataset musique \\
        --dataset_split test \\
        --retrieval_query_model_name_or_path ./predictions/train_musique/best_validation \\
        --generation_model_name meta-llama/Llama-3.1-8B-Instruct \\
        --generation_backend vllm_server \\
        --vllm_server_url http://127.0.0.1:8000/v1 \\
        --use_tta \\
        --tta_level both \\
        --tta_pseudo_label dual \\
        --tta_cross_encoder_model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --tta_inner_steps 3 \\
        --tta_query_lr 1.2 \\
        --tta_lora_rank 8 \\
        --tta_lora_lr 5e-4 \\
        --retrieval_count 8 \\
        --retrieval_buffer_size 32

Để chạy nhanh (chỉ Level 1, không cần LLM call thêm):
    python -m source.run.inference_tta \\
        ... \\
        --tta_level l1 \\
        --tta_pseudo_label ce_only
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import copy
import argparse
import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, Optional

import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.evaluation.evaluate import (
    evaluate_by_dicts,
    evaluate_multi_hop_recall_at_k,
    official_evaluate_by_dicts,
)
from source.module.generate.llama import LlamaGenerator, LlamaGeneratorConfig
from source.module.generate.vllm_server import (
    VLLMServerGenerator,
    VLLMServerGeneratorConfig,
)
from source.module.index.index import Indexer, IndexerConfig
from source.module.retrieve.cross_encoder_wrapper import CrossEncoderWrapper
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.module.retrieve.lora_utils import (
    count_lora_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
)
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState
from source.pipeline.step.end import EndStep
from source.pipeline.step.generation import (
    AnswerGenerateOutputParser,
    AnswerGeneratePromptGenerator,
    GenerationStep,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.tta_retrieval_hard import TTARetrievalStepHard
from source.utility.data_utils import clean_and_create_dir, load_data_from_jsonl
from source.utility.system_utils import seed_everything


# ──────────────────────────────────────────────────────────────────
# TeeStream (giữ nguyên từ inference.py)
# ──────────────────────────────────────────────────────────────────

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def log_hot_storage_paths(runtime_log_path, cfg):
    tracked_paths = {
        "runtime_log_dir": os.path.dirname(runtime_log_path),
        "prediction_dir": cfg.prediction_file_dir,
        "database_dir": cfg.database_path,
    }
    print("Hot storage layout:")
    for label, path in tracked_paths.items():
        real_path = os.path.realpath(path)
        location = "/docker" if real_path.startswith("/docker/") else "non-/docker"
        print(f"  {label}: path={path} realpath={real_path} [{location}]")


def describe_cuda_environment():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"Visible CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            print(
                f"Logical GPU {gpu_idx}: {props.name}, "
                f"total_memory={props.total_memory / (1024 ** 3):.2f} GiB"
            )


def summarize_tta_trace(trace_file_path: str) -> Dict[str, Any]:
    records = []
    if os.path.exists(trace_file_path):
        with open(trace_file_path, "r", encoding="utf-8") as trace_file:
            records = [json.loads(line) for line in trace_file if line.strip()]

    l1_steps = [record.get("adaptation", {}).get("l1", {}).get("steps", 0) for record in records]
    query_shifts = [record.get("adaptation", {}).get("query_shift_l2", 0.0) for record in records]
    l2_updates = [
        record.get("adaptation", {}).get("l2", {}).get("updated", False)
        for record in records
    ]

    count = len(records)
    return {
        "retrieval_hops": count,
        "unique_questions": len({record.get("question_id") for record in records}),
        "pseudo_label_failures": sum(not record.get("pseudo_label_ok", False) for record in records),
        "average_l1_steps": round(sum(l1_steps) / count, 6) if count else 0.0,
        "average_query_shift_l2": round(sum(query_shifts) / count, 6) if count else 0.0,
        "l2_update_count": sum(bool(value) for value in l2_updates),
    }


# ──────────────────────────────────────────────────────────────────
# Pipeline builder
# ──────────────────────────────────────────────────────────────────

def build_tta_pipeline(cfg, generator, retriever, indexer, cross_encoder):
    """Build pipeline với TTARetrievalStep."""
    retrieval_trace_file_path = os.path.join(
        cfg.prediction_file_dir,
        f"{cfg.dataset_split}_retrieval_trace.jsonl",
    )
    return [
        TTARetrievalStepHard(
            cfg=cfg,
            retriever=retriever,
            indexer=indexer,
            cross_encoder=cross_encoder,
            generator=generator,
            retrieval_trace_file_path=retrieval_trace_file_path,
        ),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=AnswerGeneratePromptGenerator(cfg),
            output_parser=AnswerGenerateOutputParser(cfg),
        ),
        EndStep(cfg=cfg),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=ThoughtGeneratePromptGenerator(cfg),
            output_parser=ThoughtGenerateOutputParser(cfg),
        ),
    ]


# ──────────────────────────────────────────────────────────────────
# Main run function
# ──────────────────────────────────────────────────────────────────

def run_tta(
    cfg,
    generator,
    retriever,
    indexer,
    cross_encoder,
    runtime_arguments: Optional[Dict[str, Any]] = None,
    max_inference_examples: Optional[int] = None,
):
    """Chạy TTA inference và evaluation."""
    clean_and_create_dir(cfg.prediction_file_dir)
    cfg.save()

    retrieval_trace_file_path = os.path.join(
        cfg.prediction_file_dir,
        f"{cfg.dataset_split}_retrieval_trace.jsonl",
    )
    retrieval_evaluation_file_path = os.path.join(
        cfg.prediction_file_dir,
        f"{cfg.dataset_split}_retrieval_evaluation.json",
    )
    retrieval_per_question_file_path = os.path.join(
        cfg.prediction_file_dir,
        f"{cfg.dataset_split}_retrieval_per_question.jsonl",
    )
    tta_summary_file_path = os.path.join(
        cfg.prediction_file_dir,
        f"{cfg.dataset_split}_tta_summary.json",
    )
    runtime_arguments_file_path = os.path.join(
        cfg.prediction_file_dir,
        "runtime_arguments.json",
    )
    if runtime_arguments is not None:
        with open(runtime_arguments_file_path, "w", encoding="utf-8") as f:
            json.dump(runtime_arguments, f, ensure_ascii=False, indent=4)

    inputs, id_to_ground_truths, contexts = load_data_from_jsonl(
        file_path=cfg.data_file_path,
        ground_truth_file_path=cfg.ground_truth_file_path,
        return_contexts=True,
        is_demo=cfg.demo,
    )

    if max_inference_examples is not None:
        selected_ids = list(inputs.keys())[:max_inference_examples]
        inputs = {qid: inputs[qid] for qid in selected_ids}
        id_to_ground_truths = {
            qid: id_to_ground_truths[qid] for qid in selected_ids
        }
        contexts = {qid: contexts[qid] for qid in selected_ids}
        with open(cfg.ground_truth_file_path, "w", encoding="utf-8") as f:
            json.dump(id_to_ground_truths, f, ensure_ascii=False, indent=4)
        print(f"[TTA] Limiting inference to {len(selected_ids)} examples.")

    pipeline = build_tta_pipeline(cfg, generator, retriever, indexer, cross_encoder)
    controller = PipelineController(
        pipeline=pipeline,
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )

    start_states = [
        QuestionState(question_id=qid, question=qtxt)
        for qid, qtxt in inputs.items()
    ]

    # Level 2 requires batch_size=1 for per-instance LoRA correctness
    if cfg.tta_level in ('l2', 'both') and cfg.batch_size > 1:
        print(
            f"[TTA] tta_level='{cfg.tta_level}' requires batch_size=1 for "
            f"correct per-instance LoRA isolation. "
            f"Overriding batch_size from {cfg.batch_size} → 1."
        )
        effective_batch_size = 1
    else:
        effective_batch_size = cfg.batch_size

    controller.run(start_states, batch_size=effective_batch_size)

    # ── Load results ──
    with open(cfg.ground_truth_file_path, 'r', encoding='utf-8') as f:
        id_to_ground_truths = json.load(f)
    with open(cfg.prediction_file_path, 'r', encoding='utf-8') as f:
        id_to_predictions = json.load(f)

    # ── QA Evaluation ──
    evaluation_results = evaluate_by_dicts(
        prediction_type='answer',
        id_to_ground_truths=id_to_ground_truths,
        id_to_predictions=id_to_predictions,
    )
    with open(cfg.evaluation_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f)

    # ── Retrieval MHR Evaluation ──
    try:
        retrieval_evaluation_results, retrieval_per_question = (
            evaluate_multi_hop_recall_at_k(
                contexts_by_qid=contexts,
                retrieval_trace_file_path=retrieval_trace_file_path,
                k=cfg.retrieval_count,
            )
        )
    except Exception as exc:
        retrieval_evaluation_results = {
            "retrieval_evaluation_skipped": True,
            "retrieval_evaluation_skip_reason": str(exc),
        }
        retrieval_per_question = []
        print(f"[warning] Retrieval MHR evaluation failed: {exc}")

    with open(retrieval_evaluation_file_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_evaluation_results, f, indent=4)
    with open(retrieval_per_question_file_path, 'w', encoding='utf-8') as f:
        for item in retrieval_per_question:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    tta_summary = summarize_tta_trace(retrieval_trace_file_path)
    tta_summary["tta_variant"] = "hard"
    with open(tta_summary_file_path, "w", encoding="utf-8") as f:
        json.dump(tta_summary, f, ensure_ascii=False, indent=4)

    # ── Official Evaluation ──
    try:
        official_evaluation_results = official_evaluate_by_dicts(
            prediction_type='answer',
            id_to_ground_truths=id_to_ground_truths,
            id_to_predictions=id_to_predictions,
            dataset=cfg.dataset,
        )
    except Exception as exc:
        official_evaluation_results = {
            **evaluation_results,
            "official_evaluation_skipped": True,
            "official_evaluation_skip_reason": str(exc),
        }
        print(f"[warning] Official evaluation failed: {exc}")
    with open(cfg.official_evaluation_file_path, 'w', encoding='utf-8') as f:
        json.dump(official_evaluation_results, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*60}")
    print(f"[TTA Inference Done: hard]")
    print(f"  Prediction dir:  {cfg.prediction_file_dir}")
    print(f"  EM:              {official_evaluation_results.get('em', 'N/A')}")
    print(f"  F1:              {official_evaluation_results.get('f1', 'N/A')}")
    mhr_key = f"MHR_final@{cfg.retrieval_count}"
    print(f"  {mhr_key}: {retrieval_evaluation_results.get(mhr_key, 'N/A')}")
    print(f"  TTA summary:     {tta_summary_file_path}")
    print(f"  Retrieval trace: {retrieval_trace_file_path}")
    print(f"{'='*60}\n")

    return official_evaluation_results.get('f1', 0.0)


# ──────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────

def build_arg_parser() -> ArgumentParser:
    """Build argument parser — copy từ inference.py và thêm TTA args."""
    parser = ArgumentParser(description="ReSCORE-TTA Inference")

    # ── Standard args (giữ nguyên từ inference.py) ──
    parser.add_argument("--method", type=str, default="iqatr_tta")
    parser.add_argument("--running_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--dataset",
        choices=['hotpotqa', '2wikimultihopqa', 'musique'],
        default='musique',
    )
    parser.add_argument(
        "--dataset_split", choices=['dev', 'test'], default='test'
    )
    parser.add_argument("--prompt_set", type=int, default=1)
    parser.add_argument(
        "--prompt_document_from", choices=['last_only', 'full'], default='last_only'
    )
    parser.add_argument("--prompt_max_para_count", type=int, default=15)
    parser.add_argument("--prompt_max_para_words", type=int, default=350)

    # Generator
    parser.add_argument(
        "--generation_model_name",
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
    )
    parser.add_argument(
        "--generation_backend",
        choices=["local", "vllm_server"],
        default="vllm_server",
    )
    parser.add_argument(
        "--vllm_server_url", type=str, default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--vllm_server_model", type=str, default=None)
    parser.add_argument("--vllm_server_timeout", type=float, default=600.0)
    parser.add_argument(
        "--vllm_server_max_model_len",
        type=int,
        default=None,
        help=(
            "Actual max_model_len used by the running vLLM server. "
            "When set, client prompts are clamped to this limit to avoid "
            "HTTP 400 context-length failures."
        ),
    )
    parser.add_argument(
        "--vllm_server_context_safety_margin",
        type=int,
        default=16,
        help=(
            "Extra token margin subtracted from client-side prompt limits to "
            "avoid tokenizer/counting differences with the vLLM server."
        ),
    )
    parser.add_argument(
        "--vllm_server_scoring_mode",
        choices=["prompt_logprobs", "echo_logprobs"],
        default="prompt_logprobs",
        help="Scoring mode used by vLLM server for TTA LM pseudo-labels.",
    )
    parser.add_argument(
        "--vllm_server_score_max_tokens",
        type=int,
        default=1024,
        help="Token budget for vLLM server scoring prompts.",
    )
    parser.add_argument(
        "--vllm_server_prompt_logprobs",
        type=int,
        default=20,
        help="Number of prompt logprob candidates requested from vLLM.",
    )
    parser.add_argument(
        "--vllm_server_missing_logprob_fallback",
        type=float,
        default=-20.0,
        help="Fallback logprob when vLLM omits a target token.",
    )
    parser.add_argument("--generation_max_batch_size", type=int, default=4)
    parser.add_argument("--generation_max_total_tokens", type=int, default=4096)
    parser.add_argument("--generation_max_model_len", type=int, default=4096)
    parser.add_argument("--generation_max_new_tokens", type=int, default=64)
    parser.add_argument("--generation_min_new_tokens", type=int, default=1)
    parser.add_argument("--generation_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--generation_swap_space", type=float, default=0)
    parser.add_argument("--generation_cpu_offload_gb", type=float, default=0)
    parser.add_argument(
        "--generation_enforce_eager", action="store_true"
    )
    parser.add_argument(
        "--generation_disable_custom_all_reduce", action="store_true"
    )
    parser.add_argument(
        "--generation_dtype",
        type=str,
        default="half",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
    )
    parser.add_argument("--generation_tensor_parallel_size", type=int, default=2)
    parser.add_argument(
        "--vllm_worker_multiproc_method",
        choices=['spawn', 'fork', 'forkserver'],
        default='spawn',
    )
    parser.add_argument("--disable_vllm", action='store_true')

    # Retrieval
    parser.add_argument(
        "--retrieval_count", type=int, choices=[2, 4, 6, 8], default=8
    )
    parser.add_argument(
        "--retrieval_query_type",
        choices=['last_only', 'full'],
        default='full',
    )
    parser.add_argument("--retrieval_buffer_size", type=int, default=32)
    parser.add_argument("--retrieval_no_duplicates", action='store_true')
    parser.add_argument("--retrieval_no_reasoning_sentences", action='store_true')
    parser.add_argument("--retrieval_no_wh_words", action='store_true')

    # Retriever
    parser.add_argument(
        "--retrieval_query_model_name_or_path",
        type=str,
        default='facebook/contriever-msmarco',
    )
    parser.add_argument(
        "--retrieval_passage_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument("--retrieval_batch_size", type=int, default=32)
    parser.add_argument("--retrieval_use_fp16", action='store_true')
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--database_path", type=str, default=None)

    # End
    parser.add_argument("--max_num_thought", type=int, default=6)
    parser.add_argument(
        "--answer_regex", type=str, default=".* answer is:? (.*)\\.?"
    )

    # Misc
    parser.add_argument("--demo", action='store_true')
    parser.add_argument(
        "--runtime_log_root", type=str, default="./logs/inference_tta"
    )
    parser.add_argument(
        "--prediction_root",
        type=str,
        default="./predictions",
        help="Root directory for TTA predictions and evaluation files.",
    )
    parser.add_argument(
        "--max_inference_examples",
        type=int,
        default=None,
        help="Optional example limit for smoke tests; default runs the full split.",
    )

    # ── TTA-specific args (NEW) ──────────────────────────────────
    parser.add_argument(
        "--use_tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Test-Time Adaptation (TTA).",
    )
    parser.add_argument(
        "--tta_level",
        choices=['l1', 'l2', 'both'],
        default='both',
        help=(
            "TTA adaptation level. "
            "l1: query vector optimization only (TOUR-style, no param update). "
            "l2: LoRA adapter adaptation only (requires batch_size=1). "
            "both: combine L1 + L2 (requires batch_size=1)."
        ),
    )
    parser.add_argument(
        "--tta_pseudo_label",
        choices=['ce_only', 'lm_only', 'dual'],
        default='dual',
        help=(
            "Pseudo-label type for TTA. "
            "ce_only: cross-encoder only (fastest, no extra LLM calls). "
            "lm_only: P_LM(q|d) only (no cross-encoder). "
            "dual: CE * P_LM(q|d) (best quality, needs extra LLM calls)."
        ),
    )
    parser.add_argument(
        "--tta_cross_encoder_model",
        type=str,
        default='cross-encoder/ms-marco-MiniLM-L-6-v2',
        help="HuggingFace cross-encoder model for CE pseudo-labels.",
    )
    parser.add_argument(
        "--tta_cross_encoder_batch_size", type=int, default=32
    )
    parser.add_argument(
        "--tta_cross_encoder_device",
        type=str,
        default=None,
        help="Cross-encoder device, e.g. cuda:0 or cpu. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--tta_cross_encoder_max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tta_clear_cross_encoder_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear CE cache at the start of each question to bound RAM usage.",
    )
    parser.add_argument(
        "--tta_log_every",
        type=int,
        default=1,
        help="Write one concise TTA progress line every N retrieval hops.",
    )

    # Level 1 hyperparameters (from TOUR paper)
    parser.add_argument(
        "--tta_inner_steps",
        type=int,
        default=3,
        help="Max gradient steps per hop for L1 (TOUR: max 3).",
    )
    parser.add_argument(
        "--tta_query_lr",
        type=float,
        default=1.2,
        help="Learning rate for query vector SGD (TOUR Table 7: 1.2).",
    )
    parser.add_argument(
        "--tta_momentum",
        type=float,
        default=0.99,
        help="SGD momentum for L1 (TOUR Appendix D: 0.99).",
    )
    parser.add_argument(
        "--tta_weight_decay",
        type=float,
        default=0.01,
        help="SGD weight decay for L1 (TOUR Appendix D: 0.01).",
    )
    parser.add_argument(
        "--tta_temperature",
        type=float,
        default=0.5,
        help="Temperature tau for softmax pseudo-labels (TOUR: 0.5).",
    )
    parser.add_argument(
        "--tta_nucleus_p",
        type=float,
        default=0.5,
        help="Nucleus threshold p for hard labels (TOUR: 0.5).",
    )
    parser.add_argument(
        "--tta_anchor_weight",
        type=float,
        default=0.1,
        help="Weight beta for anchor regularization ||q_t - q_0||^2.",
    )
    parser.add_argument(
        "--tta_max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for L1 query vector and L2 LoRA updates.",
    )
    parser.add_argument(
        "--tta_warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps for TOUR-style L1 query optimization.",
    )
    parser.add_argument(
        "--tta_refresh_candidates_each_step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-retrieve candidates at each L1 inner step, matching TOUR.",
    )
    parser.add_argument(
        "--tta_confidence_threshold",
        type=float,
        default=0.0,
        help="Optional pseudo-label confidence threshold; 0 disables filtering.",
    )
    parser.add_argument(
        "--tta_fail_on_pseudo_label_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast when CE/LM pseudo-labeling fails instead of silently skipping TTA.",
    )

    # Level 2 hyperparameters
    parser.add_argument(
        "--tta_lora_rank",
        type=int,
        default=8,
        help="LoRA rank r. Typical: 4, 8, 16.",
    )
    parser.add_argument(
        "--tta_lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha. scaling = alpha / rank.",
    )
    parser.add_argument(
        "--tta_lora_lr",
        type=float,
        default=5e-4,
        help="Adam learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--tta_lora_loss_weight",
        type=float,
        default=1.0,
        help="Weight alpha for the LoRA KL pseudo-label loss.",
    )
    parser.add_argument(
        "--tta_lora_num_top_layers",
        type=int,
        default=4,
        help="Number of top transformer layers to inject LoRA into.",
    )
    parser.add_argument(
        "--tta_lora_reg_weight",
        type=float,
        default=0.01,
        help="Weight gamma for LoRA norm regularization ||BA||_F^2.",
    )

    return parser


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def build_runtime_log_path(runtime_log_root, dataset, running_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = running_name or f"infer_tta_{dataset}"
    return os.path.join(
        runtime_log_root, dataset, f"{run_name}__{timestamp}", "inference_tta.log"
    )


def ensure_runtime_paths(cfg):
    required_files = [
        cfg.data_file_path,
        cfg.answer_gen_prompt_file_path,
        cfg.thought_gen_prompt_file_path,
    ]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    if not os.path.isdir(cfg.database_path):
        raise FileNotFoundError(f"Database directory not found: {cfg.database_path}")

    for artifact in ['docstore.db', 'index.faiss', 'faiss_id_to_docstore_id.pkl']:
        p = os.path.join(cfg.database_path, artifact)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing index artifact: {p}")

    # TTA-specific: verify prompt files for dual/lm_only
    if cfg.tta_pseudo_label in ('dual', 'lm_only'):
        for p in [cfg.tta_q_rel_input_prompt_file_path,
                  cfg.tta_q_rel_output_prompt_file_path]:
            if not os.path.exists(p):
                print(
                    f"[inference_tta] WARNING: TTA prompt not found: {p}. "
                    f"Dual pseudo-label requires q_rel prompts. "
                    f"Will fallback to ce_only at runtime."
                )


if __name__ == '__main__':
    parser = build_arg_parser()
    opt = parser.parse_args()

    if not opt.use_tta:
        parser.error("inference_tta.py requires TTA; use inference.py for non-TTA inference.")
    if opt.max_inference_examples is not None and opt.max_inference_examples <= 0:
        parser.error("--max_inference_examples must be greater than zero.")
    if opt.tta_log_every <= 0:
        parser.error("--tta_log_every must be greater than zero.")
    if opt.running_name is None:
        opt.running_name = (
            f"infer_tta_hard_{opt.dataset}_{opt.tta_level}_{opt.tta_pseudo_label}"
        )

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = opt.vllm_worker_multiproc_method
    try:
        mp.set_start_method(opt.vllm_worker_multiproc_method, force=True)
    except RuntimeError:
        pass

    seed_everything(opt.seed)

    # ── Build PipelineConfig ──
    from dataclasses import fields as dc_fields
    cfg_field_names = {f.name for f in dc_fields(PipelineConfig)}

    # Map opt attrs to cfg kwargs (only matching fields)
    cfg_kwargs = {
        k: v for k, v in vars(opt).items()
        if k in cfg_field_names
    }

    # TTA fields — set explicitly (not in standard PipelineConfig fields yet)
    cfg = PipelineConfig(**cfg_kwargs)
    if opt.database_path:
        cfg.database_path_override = opt.database_path
    elif opt.retrieval_passage_model_name_or_path is None:
        cfg.database_path_override = os.path.join(
            "./data/database/contriever_msmarco",
            opt.dataset,
        )
    cfg.prediction_root_override = opt.prediction_root

    # Set TTA fields (added by config_patch.py)
    for tta_attr in [
        'use_tta', 'tta_level', 'tta_pseudo_label',
        'tta_cross_encoder_model', 'tta_cross_encoder_batch_size',
        'tta_cross_encoder_device', 'tta_cross_encoder_max_length',
        'tta_clear_cross_encoder_cache', 'tta_log_every',
        'tta_inner_steps', 'tta_query_lr', 'tta_momentum',
        'tta_weight_decay', 'tta_temperature', 'tta_nucleus_p',
        'tta_anchor_weight', 'tta_max_grad_norm', 'tta_warmup_steps',
        'tta_refresh_candidates_each_step', 'tta_confidence_threshold',
        'tta_fail_on_pseudo_label_error', 'tta_lora_rank', 'tta_lora_alpha',
        'tta_lora_lr', 'tta_lora_loss_weight',
        'tta_lora_num_top_layers', 'tta_lora_reg_weight',
    ]:
        if hasattr(cfg, tta_attr) and hasattr(opt, tta_attr):
            setattr(cfg, tta_attr, getattr(opt, tta_attr))

    # ── Runtime logging ──
    runtime_log_path = build_runtime_log_path(
        opt.runtime_log_root, cfg.dataset, cfg.running_name
    )
    os.makedirs(os.path.dirname(runtime_log_path), exist_ok=True)
    runtime_arguments = vars(opt).copy()
    with open(
        os.path.join(os.path.dirname(runtime_log_path), "runtime_arguments.json"),
        "w",
        encoding="utf-8",
    ) as runtime_arguments_file:
        json.dump(runtime_arguments, runtime_arguments_file, ensure_ascii=False, indent=4)

    with open(runtime_log_path, "a", encoding="utf-8", buffering=1) as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

        try:
            print(f"Runtime log: {runtime_log_path}")
            print(f"Run name: {cfg.running_name}")
            print(f"Dataset: {cfg.dataset} / {cfg.dataset_split}")
            print(f"Data file: {cfg.data_file_path}")
            print(f"Database path: {cfg.database_path}")
            print(f"Prediction directory: {cfg.prediction_file_dir}")
            print(f"TTA level: {opt.tta_level}")
            print(f"TTA pseudo-label: {opt.tta_pseudo_label}")
            print(
                "TTA optimization: "
                f"inner_steps={opt.tta_inner_steps}, "
                f"query_lr={opt.tta_query_lr}, "
                f"anchor_weight={opt.tta_anchor_weight}, "
                f"max_grad_norm={opt.tta_max_grad_norm}, "
                f"refresh_candidates_each_step={opt.tta_refresh_candidates_each_step}, "
                f"lora_rank={opt.tta_lora_rank}, "
                f"lora_lr={opt.tta_lora_lr}, "
                f"lora_loss_weight={opt.tta_lora_loss_weight}, "
                f"lora_reg_weight={opt.tta_lora_reg_weight}"
            )
            log_hot_storage_paths(runtime_log_path, cfg)
            ensure_runtime_paths(cfg)
            describe_cuda_environment()

            # ── Generator ──
            if opt.generation_backend == "vllm_server":
                server_max_total_tokens = opt.generation_max_total_tokens
                if opt.vllm_server_max_model_len is not None:
                    server_max_total_tokens = min(
                        server_max_total_tokens,
                        opt.vllm_server_max_model_len,
                    )
                server_score_max_tokens = min(
                    opt.vllm_server_score_max_tokens,
                    max(2, server_max_total_tokens - 1),
                )
                generator = VLLMServerGenerator(
                    VLLMServerGeneratorConfig(
                        model_name=opt.generation_model_name,
                        served_model_name=opt.vllm_server_model,
                        base_url=opt.vllm_server_url,
                        batch_size=opt.generation_max_batch_size,
                        max_total_tokens=server_max_total_tokens,
                        score_max_tokens=server_score_max_tokens,
                        max_new_tokens=opt.generation_max_new_tokens,
                        min_new_tokens=opt.generation_min_new_tokens,
                        request_timeout=opt.vllm_server_timeout,
                        scoring_mode=opt.vllm_server_scoring_mode,
                        prompt_logprobs=opt.vllm_server_prompt_logprobs,
                        missing_logprob_fallback=(
                            opt.vllm_server_missing_logprob_fallback
                        ),
                        context_safety_margin=(
                            opt.vllm_server_context_safety_margin
                        ),
                    )
                )
                print(
                    f"Generator: vllm_server @ {opt.vllm_server_url}, "
                    f"scoring_mode={opt.vllm_server_scoring_mode}, "
                    f"server_max_model_len={opt.vllm_server_max_model_len}, "
                    f"max_total_tokens={server_max_total_tokens}, "
                    f"score_max_tokens={server_score_max_tokens}, "
                    f"context_safety_margin={opt.vllm_server_context_safety_margin}, "
                    f"prompt_logprobs={opt.vllm_server_prompt_logprobs}"
                )
            else:
                generator = LlamaGenerator(
                    LlamaGeneratorConfig(
                        model_name=opt.generation_model_name,
                        batch_size=opt.generation_max_batch_size,
                        max_total_tokens=opt.generation_max_total_tokens,
                        max_model_len=opt.generation_max_model_len,
                        max_new_tokens=opt.generation_max_new_tokens,
                        min_new_tokens=opt.generation_min_new_tokens,
                        gpu_memory_utilization=opt.generation_gpu_memory_utilization,
                        swap_space=opt.generation_swap_space,
                        cpu_offload_gb=opt.generation_cpu_offload_gb,
                        dtype=opt.generation_dtype,
                        use_vllm=not opt.disable_vllm,
                        tensor_parallel_size=opt.generation_tensor_parallel_size,
                        enforce_eager=opt.generation_enforce_eager,
                        disable_custom_all_reduce=opt.generation_disable_custom_all_reduce,
                    )
                )
                print(f"Generator: local, model={opt.generation_model_name}")

            # ── Retriever ──
            retriever = DenseRetriever(
                DenseRetrieverConfig(
                    query_model_name_or_path=opt.retrieval_query_model_name_or_path,
                    passage_model_name_or_path=opt.retrieval_passage_model_name_or_path,
                    batch_size=opt.retrieval_batch_size,
                    training_strategy=None,   # inference mode — no grad by default
                    use_fp16=opt.retrieval_use_fp16,
                    device=opt.retriever_device,
                )
            )
            print(f"Retriever: {opt.retrieval_query_model_name_or_path}")

            # ── Inject LoRA (Level 2) ──
            if opt.tta_level in ('l2', 'both'):
                retriever.query_model = inject_lora(
                    model=retriever.query_model,
                    rank=opt.tta_lora_rank,
                    lora_alpha=opt.tta_lora_alpha,
                    num_top_layers=opt.tta_lora_num_top_layers,
                )
                trainable_lora_params = mark_only_lora_as_trainable(
                    retriever.query_model
                )
                retriever.query_model.eval()
                lora_stats = count_lora_parameters(retriever.query_model)
                print(
                    f"[TTA] LoRA injected: rank={opt.tta_lora_rank}, "
                    f"alpha={opt.tta_lora_alpha}, "
                    f"top_{opt.tta_lora_num_top_layers}_layers. "
                    f"LoRA params: {lora_stats['lora']:,} "
                    f"({lora_stats['lora_pct']:.2f}% of total), "
                    f"trainable={trainable_lora_params:,}"
                )

            # ── Cross-encoder ──
            if opt.tta_pseudo_label in ('ce_only', 'dual'):
                cross_encoder = CrossEncoderWrapper(
                    model_name_or_path=opt.tta_cross_encoder_model,
                    device=opt.tta_cross_encoder_device,
                    max_length=opt.tta_cross_encoder_max_length,
                    batch_size=opt.tta_cross_encoder_batch_size,
                )
                print(
                    f"[TTA] Cross-encoder: {opt.tta_cross_encoder_model}, "
                    f"device={cross_encoder.device}, "
                    f"batch_size={opt.tta_cross_encoder_batch_size}, "
                    f"max_length={opt.tta_cross_encoder_max_length}"
                )
            else:
                cross_encoder = None
                print("[TTA] Cross-encoder disabled for lm_only pseudo-labels.")

            # ── Indexer ──
            indexer = Indexer.load_local(
                IndexerConfig(
                    embedding_sz=768,
                    database_path=cfg.database_path,
                )
            )

            # ── Run TTA inference ──
            run_tta(
                cfg,
                generator,
                retriever,
                indexer,
                cross_encoder,
                runtime_arguments=runtime_arguments,
                max_inference_examples=opt.max_inference_examples,
            )

        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
