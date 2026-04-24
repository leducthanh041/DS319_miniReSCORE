import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3,4,5")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import sys
from dataclasses import fields
from datetime import datetime
from typing import List

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReSCORE retriever")

    parser.add_argument("--method", type=str, default="rescore", help="Method name")
    parser.add_argument("--running_name", type=str, default=None, help="Name for the run")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
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
        "--generation_max_batch_size",
        type=int,
        default=1,
        help="Batch size for generation/scoring",
    )
    parser.add_argument(
        "--generation_max_total_tokens",
        type=int,
        default=3072,
        help="Max total tokens for generation/scoring",
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
        default="9GiB",
        help="Max memory per visible GPU for generator when device_map=auto",
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
    parser.add_argument("--retrieval_batch_size", type=int, default=16, help="Retriever embedding batch size")
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
        default=None,
        help="Retriever device, for example cuda:3 or cpu",
    )

    parser.add_argument("--max_num_thought", type=int, default=6, help="Max number of thoughts")
    parser.add_argument("--answer_regex", type=str, default=".* Answer: <.*>\\.?", help="Answer regex")
    parser.add_argument("--match_all_on_failure", action="store_true", help="Regex fallback")
    parser.add_argument("--demo", action="store_true", help="Use demo subset")

    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--temperature_r", type=float, default=0.1, help="Retriever temperature")
    parser.add_argument("--temperature_lm", type=float, default=1.0, help="LM temperature")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")

    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--validation_freq", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--save_freq", type=int, default=100, help="Save retriever every N steps")
    parser.add_argument("--runtime_log_root", type=str, default="./logs/train", help="Runtime log root directory")

    return parser.parse_args()


def build_pipeline_config(args):
    cfg_field_names = {field.name for field in fields(PipelineConfig)}
    cfg_kwargs = {
        key: value for key, value in vars(args).items()
        if key in cfg_field_names
    }
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

    if any(
        not (os.path.realpath(path).startswith("/docker/") or os.path.realpath(path) == "/docker")
        for path in tracked_paths.values()
    ):
        print(
            "[warning] Hot write paths are not fully on /docker. "
            "To avoid NFS-induced D-state hangs, run: bash script/setup_local_hot_data.sh"
        )


def reset_controller_state(controller):
    controller.state_tree.clear()
    controller.running_state_ids.clear()
    controller.end_state_ids.clear()


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


def validate(cfg, controller, epoch, num_steps, demo, num_workers):
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
    dev_dataloader = build_dataloader(dev_start_states, cfg.batch_size, num_workers)

    with torch.no_grad():
        for batch in dev_dataloader:
            batch_loss = controller.train(batch)
            total_loss += batch_loss.item()
            total_batches += 1
            reset_controller_state(controller)

    avg_loss = total_loss / max(total_batches, 1)
    print(f"[validation] epoch={epoch} step={num_steps} avg_loss={avg_loss:.6f}")

    retriever.query_model.train(query_model_was_training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    for epoch in range(cfg.n_epochs):
        num_steps = 0
        print(f"[epoch-start] epoch={epoch}")

        for batch in dataloader:
            num_steps += 1
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
                validate(cfg, controller, epoch, num_steps, cfg.demo, args.num_workers)

            if args.save_freq > 0 and num_steps % args.save_freq == 0:
                save_path = os.path.join(cfg.prediction_file_dir, f"epoch_{epoch}_step_{num_steps}")
                retriever.query_model.save_pretrained(save_path)
                retriever.query_tokenizer.save_pretrained(save_path)
                print(f"[checkpoint] saved retriever to {save_path}")

            if torch.cuda.is_available() and num_steps % 20 == 0:
                torch.cuda.empty_cache()

        if num_accumulations > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            num_accumulations = 0
            print(f"[optimizer] flushed trailing gradients at epoch={epoch}")

        print(f"[epoch-end] epoch={epoch} optimizer_steps={optimizer_steps}")

    save_path = cfg.prediction_file_dir
    retriever.query_model.save_pretrained(save_path)
    retriever.query_tokenizer.save_pretrained(save_path)
    print(f"[done] final retriever saved to {save_path}")


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

            print(
                "Generator config: "
                f"model={cfg.generation_model_name}, "
                f"device_map={generation_device_map}, "
                f"generator_gpu={args.generator_gpu}, "
                f"max_memory_per_gpu={args.generation_max_memory_per_gpu}"
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

            generator = LlamaGenerator(
                LlamaGeneratorConfig(
                    model_name=cfg.generation_model_name,
                    batch_size=cfg.generation_max_batch_size,
                    max_total_tokens=cfg.generation_max_total_tokens,
                    max_new_tokens=cfg.generation_max_new_tokens,
                    min_new_tokens=cfg.generation_min_new_tokens,
                    use_vllm=False,
                    gpu=args.generator_gpu,
                    device_map=generation_device_map,
                    max_memory_per_gpu=args.generation_max_memory_per_gpu,
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
