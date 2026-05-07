# ReSCORE

> Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision

This repository is an active reproduction and implementation workspace for the ReSCORE paper. The code under `source/` is the execution source of truth.

Project page: <https://leeds1219.github.io/ReSCORE>  
Paper: <https://arxiv.org/abs/2505.21250>

## Status

This repo is still under active development. Paths, defaults, and scripts may change while reproduction is being stabilized.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The current runtime stack includes:

- `vllm==0.6.3.post1`
- `faiss-gpu==1.7.2`
- `accelerate==1.1.0`
- `deepspeed==0.15.3`
- `wandb`

If you use gated Hugging Face checkpoints such as `meta-llama/Llama-3.1-8B-Instruct`, set your Hugging Face token before launching training or inference.

## Data Preparation

The expected workflow is:

1. download raw multi-hop QA data
2. preprocess it into processed JSONL files
3. build the retrieval database and index

### Recommended storage layout

To reduce I/O pressure and avoid stalls, keep hot-write paths on Docker storage by default:

- `logs` -> Docker
- `data/database` -> Docker

If the machine has enough local disk space and strong storage bandwidth, you can keep these paths in the current repo directory instead. Use whichever layout matches your resource budget.

### Raw data download

```bash
bash script/download/multihop_raw_data.sh
```

### Processed data generation

```bash
bash script/download/multihop_processed_data.sh
```

This prepares processed data for:

- `hotpotqa`
- `2wikimultihopqa`
- `musique`

### Retrieval DB build

```bash
bash script/download/build.sh
```

This script handles preprocessing, embedding generation, and index building. Build logs are written under `logs/build/...`.

## Training

Main entry point:

```bash
python -m source.run.train
```

Example command for MuSiQue:

```bash
CUDA_VISIBLE_DEVICES=5,6 python -m source.run.train \
  --running_name train_musique \
  --dataset musique \
  --method rescore \
  --prompt_set 1 \
  --retrieval_query_model_name_or_path facebook/contriever-msmarco \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0
```

Example command for 2WikiMultiHopQA:

```bash
CUDA_VISIBLE_DEVICES=5,6 python -m source.run.train \
  --running_name train_2wikimultihopqa \
  --dataset 2wikimultihopqa \
  --method rescore \
  --prompt_set 1 \
  --retrieval_query_model_name_or_path facebook/contriever-msmarco \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0
```

Current training defaults in `source/run/train.py`:

- `generation_use_vllm=True`
- `generation_dtype=half`
- `generation_tensor_parallel_size=2`
- `generation_gpu_memory_utilization=0.95`
- `generation_max_total_tokens=4096`
- `generation_max_model_len=4096`
- `retrieval_count=8`
- `retrieval_buffer_size=32`
- `retrieval_batch_size=32`
- `max_num_thought=6`
- `n_epochs=3`
- `validation_freq=10`
- `save_freq=10`

Training logs are written to `logs/train/...`.

Prediction artifacts and checkpoints are written under `predictions/...`.

## Inference

Main entry point:

```bash
python -m source.run.inference
```

Example command for MuSiQue inference with the best local checkpoint:

```bash
CUDA_VISIBLE_DEVICES=5,6 python -m source.run.inference \
  --method rescore \
  --running_name infer_musique_best \
  --dataset musique \
  --dataset_split test \
  --prompt_set 1 \
  --retrieval_query_model_name_or_path ./predictions/musique/train_musique_resume_best___llama_3.1_8b_instruct___best_validation/multi_retrieval___train/prompt_set__1/retr_count__4/epoch_0_step_690 \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0
```

Current inference defaults in `source/run/inference.py`:

- `generation_use_vllm=True`
- `generation_dtype=half`
- `generation_tensor_parallel_size=2`
- `generation_gpu_memory_utilization=0.95`
- `generation_max_total_tokens=4096`
- `generation_max_model_len=4096`
- `generation_max_batch_size=4`
- `retrieval_count=8`
- `retrieval_buffer_size=32`
- `retrieval_batch_size=32`
- `max_num_thought=6`

Inference logs are written to `logs/inference/...`.

Evaluation artifacts are written next to the inference prediction directory.

## Storage Policy

Recommended default:

- keep `logs` on Docker
- keep `data/database` on Docker
- keep `predictions` in the current repo directory if you want easier checkpoint access

If the machine is strong enough and local disk is fast, you can move more paths back to the current directory. If I/O becomes the bottleneck, move the hot-write paths back to Docker first.

## Quick Start

1. Install dependencies with `pip install -r requirements.txt`
2. Download raw data with `bash script/download/multihop_raw_data.sh`
3. Preprocess data with `bash script/download/multihop_processed_data.sh`
4. Build the retrieval DB with `bash script/download/build.sh`
5. Train with `python -m source.run.train`
6. Run inference with `python -m source.run.inference`

## Citation

```bibtex
@inproceedings{lee-etal-2025-rescore,
  title = "{R}e{SCORE}: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision",
  author = "Lee, Dosung and Oh, Wonjun and Kim, Boyoung and Kim, Minyoung and Park, Joonsuk and Seo, Paul Hongsuck",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025"
}
```
