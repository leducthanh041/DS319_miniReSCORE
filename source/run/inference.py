import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5,6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from datetime import datetime
import sys

import torch

from source.utility.data_utils import (
    load_data_from_jsonl
)
from source.pipeline.step.retrieval import RetrievalStep
from source.pipeline.step.generation import (
    GenerationStep, 
    AnswerGenerateOutputParser, 
    AnswerGeneratePromptGenerator,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.end import EndStep

from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState
from source.utility.system_utils import seed_everything
from source.utility.data_utils import clean_and_create_dir

from source.module.generate.llama import (
    LlamaGenerator,
    LlamaGeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)
import json
from source.evaluation.evaluate import (
    evaluate_by_dicts,
    official_evaluate_by_dicts,
)
import copy


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


def build_runtime_log_path(runtime_log_root, dataset, running_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = running_name or f"infer_{dataset}"
    return os.path.join(runtime_log_root, dataset, f"{run_name}__{timestamp}", "inference.log")


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

    database_artifacts = [
        os.path.join(cfg.database_path, "docstore.db"),
        os.path.join(cfg.database_path, "index.faiss"),
        os.path.join(cfg.database_path, "faiss_id_to_docstore_id.pkl"),
    ]
    missing_artifacts = [path for path in database_artifacts if not os.path.exists(path)]
    if missing_artifacts:
        raise FileNotFoundError("Missing index artifacts:\n" + "\n".join(missing_artifacts))


def describe_cuda_environment():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"Visible CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for gpu_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_idx)
            total_gib = props.total_memory / (1024 ** 3)
            print(f"Logical GPU {gpu_idx}: {props.name}, total_memory={total_gib:.2f} GiB")


def run(cfg, generator, retriever, indexer):
    clean_and_create_dir(cfg.prediction_file_dir)
    cfg.save()

    inputs, id_to_ground_truths, contexts = load_data_from_jsonl(
        file_path = cfg.data_file_path,
        ground_truth_file_path=cfg.ground_truth_file_path,
        return_contexts=True,
        is_demo=cfg.demo,
    )

    pipeline = [
        RetrievalStep(
            cfg=cfg,
            retriever=retriever,
            indexer=indexer,
        ),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=AnswerGeneratePromptGenerator(cfg),
            output_parser=AnswerGenerateOutputParser(cfg)
        ),
        EndStep(
            cfg=cfg,
        ),
        GenerationStep(
            cfg=cfg,
            generator=generator,
            prompt_generator=ThoughtGeneratePromptGenerator(cfg),
            output_parser=ThoughtGenerateOutputParser(cfg)
        ),
    ]
    controller = PipelineController(
        pipeline=pipeline,
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )
    start_states = [
        QuestionState(
            question_id=question_id,
            question=question_text
        )
        for question_id, question_text in inputs.items()
    ]
    controller.run(
        start_states,
        batch_size=cfg.batch_size
    )
    
    with open(cfg.ground_truth_file_path, 'r', encoding='utf-8') as f:
        id_to_ground_truths = json.load(f)
        
    with open(cfg.prediction_file_path, 'r', encoding='utf-8') as f:
        id_to_predictions = json.load(f)
        
    evaluation_results = evaluate_by_dicts(
        prediction_type='answer',
        id_to_ground_truths=id_to_ground_truths,
        id_to_predictions=id_to_predictions,
    )
    with open(cfg.evaluation_file_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f)
    official_evaluation_results = official_evaluate_by_dicts(
        prediction_type='answer',
        id_to_ground_truths=id_to_ground_truths,
        id_to_predictions=id_to_predictions,
        dataset=cfg.dataset
    )
    with open(cfg.official_evaluation_file_path, 'w') as f:
        json.dump(official_evaluation_results, f)

    print(f"Prediction directory: {cfg.prediction_file_dir}")
    print(f"Prediction JSON: {cfg.prediction_file_path}")
    print(f"Evaluation JSON: {cfg.evaluation_file_path}")
    print(f"Official evaluation JSON: {cfg.official_evaluation_file_path}")
    print(f"[done] official_f1={official_evaluation_results['f1']}")
    
    return official_evaluation_results['f1']


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # General parameters
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="iqatr or base"
    )    
    parser.add_argument(
        "--running_name",
        type=str,
        default=None,
        help=""
    )    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference Batch size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed"
    )
    parser.add_argument(
        "--dataset",
        choices=['hotpotqa', '2wikimultihopqa', 'musique'],
        default='musique',
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset_split",
        choices=['dev', 'test'],
        default='test',
        help="Dataset split for inference"
    )
    parser.add_argument(
        "--prompt_set",
        type=int,
        default=1,
        help="prompt_set"
    )
    parser.add_argument(
        "--prompt_document_from",
        choices=['last_only', 'full'],
        default='last_only',
        help="prompt_document_from"
    )
    parser.add_argument(
        "--prompt_max_para_count",
        type=int,
        default=15,
        help="Maximum number of paragraphs in prompt"
    )
    parser.add_argument(
        "--prompt_max_para_words",
        type=int,
        default=350,
        help="Maximum words per paragraph in prompt"
    )

    # Generator
    parser.add_argument(
        "--generation_model_name",
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help="Generation model name"
    )
    parser.add_argument(
        "--generation_max_batch_size",
        type=int,
        default=4,
        help="Maximum batch size for generation"
    )
    parser.add_argument(
        "--generation_max_total_tokens",
        type=int,
        default=4096,
        help="Maximum total tokens for generation"
    )
    parser.add_argument(
        "--generation_max_model_len",
        type=int,
        default=4096,
        help="vLLM max_model_len/KV-cache length. Lower this on 11GB GPUs, for example 2048 or 1024."
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--generation_min_new_tokens",
        type=int,
        default=1,
        help="Minimum new tokens for generation"
    )
    parser.add_argument(
        "--generation_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM GPU memory utilization"
    )
    parser.add_argument(
        "--generation_swap_space",
        type=float,
        default=0,
        help="vLLM CPU swap space in GiB per GPU"
    )
    parser.add_argument(
        "--generation_cpu_offload_gb",
        type=float,
        default=0,
        help="vLLM CPU offload size in GiB"
    )
    parser.add_argument(
        "--generation_dtype",
        type=str,
        default="half",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="vLLM model dtype. Use half/float16 for RTX 2080 Ti because it does not support bfloat16."
    )
    parser.add_argument(
        "--generation_tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size; defaults to all visible GPUs"
    )
    parser.add_argument(
        "--vllm_worker_multiproc_method",
        choices=['spawn', 'fork', 'forkserver'],
        default='spawn',
        help="vLLM multiprocessing start method; spawn avoids CUDA fork re-init errors"
    )
    parser.add_argument(
        "--disable_vllm",
        action='store_true',
        help="Debug fallback: use Hugging Face Transformers instead of vLLM"
    )

    # Retrieval
    parser.add_argument(
        "--retrieval_count",
        type=int,
        choices=[2, 4, 6, 8],
        default=8,
        help="Number of retrievals"
    )
    parser.add_argument(
        "--retrieval_query_type",
        choices=['last_only', 'full'],
        default='full',
        help="Retrieval Query type"
    )
    parser.add_argument(
        "--retrieval_buffer_size",
        type=int,
        default=32,
        help="Retrieval buffer size"
    )
    parser.add_argument(
        "--retrieval_no_duplicates",
        action='store_true',
        help="Remove duplicate retrievals"
    )
    parser.add_argument(
        "--retrieval_no_reasoning_sentences",
        action='store_true',
        help="Exclude reasoning sentences from retrieval"
    )
    parser.add_argument(
        "--retrieval_no_wh_words",
        action='store_true',
        help="Exclude WH-words from retrieval"
    )

    # Retriever
    parser.add_argument(
        "--retrieval_query_model_name_or_path",
        type=str,
        default='facebook/contriever-msmarco',
        help="Query model name or path for retrieval"
    )
    parser.add_argument(
        "--retrieval_passage_model_name_or_path",
        type=str,
        default=None,
        help="Passage model name or path for retrieval"
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=32,
        help="Batch size for retrieval"
    )
    parser.add_argument(
        "--retrieval_training_strategy",
        choices=['query_only', 'both'],
        default=None,
        help="Training strategy for retrieval"
    )
    parser.add_argument(
        "--retrieval_use_fp16",
        action='store_true',
        help="Use FP16 for retrieval"
    )
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Retriever device, for example cuda:1 or cpu"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Override retrieval DB directory containing docstore.db, index.faiss, and faiss_id_to_docstore_id.pkl"
    )

    # End
    parser.add_argument(
        "--max_num_thought",
        type=int,
        default=6,
        help="Maximum number of thoughts"
    )
    parser.add_argument(
        "--answer_regex",
        type=str,
        default=".* answer is:? (.*)\\.?",
        help="Regex pattern to extract answer"
    )
    
    # Etc
    parser.add_argument(
        "--demo",
        action='store_true',
        help="Whether to use Demo"
    )
    parser.add_argument(
        "--runtime_log_root",
        type=str,
        default="./logs/inference",
        help="Runtime log root directory"
    )

    opt = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = opt.vllm_worker_multiproc_method
    try:
        mp.set_start_method(opt.vllm_worker_multiproc_method, force=True)
    except RuntimeError:
        pass

    cfg_kwargs = vars(opt).copy()
    database_path = cfg_kwargs.pop("database_path")
    runtime_log_root = cfg_kwargs.pop("runtime_log_root")
    retriever_device = cfg_kwargs.pop("retriever_device")
    generation_max_model_len = cfg_kwargs.pop("generation_max_model_len")
    generation_gpu_memory_utilization = cfg_kwargs.pop("generation_gpu_memory_utilization")
    generation_swap_space = cfg_kwargs.pop("generation_swap_space")
    generation_cpu_offload_gb = cfg_kwargs.pop("generation_cpu_offload_gb")
    generation_dtype = cfg_kwargs.pop("generation_dtype")
    generation_tensor_parallel_size = cfg_kwargs.pop("generation_tensor_parallel_size")
    cfg_kwargs.pop("vllm_worker_multiproc_method")
    disable_vllm = cfg_kwargs.pop("disable_vllm")

    seed_everything(opt.seed)
    cfg = PipelineConfig(**cfg_kwargs)
    if database_path:
        cfg.database_path_override = database_path

    runtime_log_path = build_runtime_log_path(runtime_log_root, cfg.dataset, cfg.running_name)
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
            print(f"vLLM worker multiprocessing method: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")
            log_hot_storage_paths(runtime_log_path, cfg)
            ensure_runtime_paths(cfg)
            describe_cuda_environment()
            print(
                "Generator config: "
                f"model={cfg.generation_model_name}, "
                f"use_vllm={not disable_vllm}, "
                f"max_batch_size={cfg.generation_max_batch_size}, "
                f"max_total_tokens={cfg.generation_max_total_tokens}, "
                f"max_model_len={generation_max_model_len or cfg.generation_max_total_tokens}, "
                f"gpu_memory_utilization={generation_gpu_memory_utilization}, "
                f"swap_space={generation_swap_space}, "
                f"cpu_offload_gb={generation_cpu_offload_gb}, "
                f"dtype={generation_dtype}, "
                f"tensor_parallel_size={generation_tensor_parallel_size or 'all_visible'}"
            )
            print(
                "Retriever config: "
                f"model={cfg.retrieval_query_model_name_or_path}, "
                f"passage_model={cfg.retrieval_passage_model_name_or_path}, "
                f"device={retriever_device}, "
                f"fp16={cfg.retrieval_use_fp16}, "
                f"batch_size={cfg.retrieval_batch_size}"
            )

            generator = LlamaGenerator(
                LlamaGeneratorConfig(
                    model_name=opt.generation_model_name,
                    batch_size=opt.generation_max_batch_size,
                    max_total_tokens=opt.generation_max_total_tokens,
                    max_model_len=generation_max_model_len,
                    max_new_tokens=opt.generation_max_new_tokens,
                    min_new_tokens=opt.generation_min_new_tokens,
                    gpu_memory_utilization=generation_gpu_memory_utilization,
                    swap_space=generation_swap_space,
                    cpu_offload_gb=generation_cpu_offload_gb,
                    dtype=generation_dtype,
                    use_vllm=not disable_vllm,
                    tensor_parallel_size=generation_tensor_parallel_size,
                )
            )
            retriever = DenseRetriever(
                DenseRetrieverConfig(
                    query_model_name_or_path=opt.retrieval_query_model_name_or_path,
                    passage_model_name_or_path=opt.retrieval_passage_model_name_or_path,
                    batch_size=opt.retrieval_batch_size,
                    training_strategy=opt.retrieval_training_strategy,
                    use_fp16=opt.retrieval_use_fp16,
                    device=retriever_device,
                )
            )
            indexer = Indexer.load_local(
                IndexerConfig(
                    embedding_sz=768,
                    database_path=cfg.database_path
                )
            )

            run(cfg, generator, retriever, indexer)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
