# ReSCORE Reproduction Workspace

This repository is a practical reproduction workspace for ReSCORE, inspired by the original project at <https://github.com/leeds1219/ReSCORE>.

Paper: <https://arxiv.org/abs/2505.21250>  
Project page: <https://leeds1219.github.io/ReSCORE>

The active execution path in this repo is under `source/`. Scripts in `demo/` or older notes may not match the current training and inference path.

## What This Repo Runs

ReSCORE trains a dense retriever for multi-hop QA with relevance-consistency supervision from an LLM. In this reproduction setup:

- the retriever is trained with gradients on one GPU
- Llama generation and scoring are served by a persistent vLLM server on separate GPUs
- retrieval databases are FAISS/docstore artifacts under `data/database`
- logs are written to `logs/...`
- checkpoints and predictions are written to `predictions/...`

Supported datasets in the active scripts:

- `hotpotqa`
- `2wikimultihopqa`
- `musique`

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Core runtime packages:

- `vllm==0.6.3.post1`
- `faiss-gpu==1.7.2`
- `accelerate==1.1.0`
- `deepspeed==0.15.3`
- `wandb`

For Llama 3.1 checkpoints, configure Hugging Face access first:

```bash
export HF_TOKEN=<your_huggingface_token>
```

If you use a fine-grained token, enable access to public gated repositories in the Hugging Face token settings.

## Storage Layout

This project can produce heavy logs, databases, and checkpoints. On shared NFS storage, SQLite/FAISS/log writes can cause I/O stalls. The recommended layout is:

- keep source code in this repo
- keep `logs` on local Docker/SSD storage
- keep `data/database` on local Docker/SSD storage
- keep `predictions` local to the repo if you want easy checkpoint access

Example symlink layout:

```bash
mkdir -p /docker/data/$USER/ReSCORE/logs
mkdir -p /docker/data/$USER/ReSCORE/data/database

ln -s /docker/data/$USER/ReSCORE/logs logs
ln -s /docker/data/$USER/ReSCORE/data/database data/database
```

If your local disk is fast enough and has enough space, you can keep all paths in the current repo instead.

## Data Preparation

Download raw multi-hop datasets:

```bash
bash script/download/multihop_raw_data.sh
```

Generate processed QA files:

```bash
bash script/download/multihop_processed_data.sh
```

Build the retrieval database and FAISS index:

```bash
bash script/download/build.sh
```

Expected retrieval DB layout for each dataset:

```text
data/database/contriever_msmarco/<dataset>/
  docstore.db
  index.faiss
  faiss_id_to_docstore_id.pkl
```

Expected processed data layout:

```text
data/processed_data/<dataset>/
  train.jsonl
  dev.jsonl
  test.jsonl
```

## vLLM Server

Training uses a persistent vLLM server so the Llama model is loaded once and reused across runs. This avoids repeatedly loading the LLM inside `train.py`.

Recommended GPU split:

- GPU `1`: retriever training
- GPU `5,6`: vLLM server for Llama generation/scoring

Start vLLM before training:

```bash
python script/preload_vllm_server.py \
  --cuda_visible_devices 5,6 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor_parallel_size 2 \
  --dtype half \
  --gpu_memory_utilization 0.9 \
  --max_model_len 4096 \
  --max_num_seqs 20 \
  --swap_space 0 \
  --cpu_offload_gb 0 \
  --enforce_eager \
  --port 8000
```

Check that the server is alive:

```bash
curl http://127.0.0.1:8000/v1/models
```

The server script intentionally does not kill existing vLLM processes. Use `script/kill_vllm_processes.py` manually if you want to stop them.

## Training

Main entry point:

```bash
python -m source.run.train
```

Use the same ReSCORE training template for all three datasets. Set `DATASET` to one of:

```text
hotpotqa
2wikimultihopqa
musique
```

Then run:

```bash
DATASET=2wikimultihopqa
RUNNING_NAME=train_${DATASET}_vllm_server_from_scratch

CUDA_VISIBLE_DEVICES=1 python -m source.run.train \
  --running_name "${RUNNING_NAME}" \
  --dataset "${DATASET}" \
  --method rescore \
  --prompt_set 1 \
  --retrieval_query_model_name_or_path facebook/contriever-msmarco \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0 \
  --generation_backend vllm_server \
  --vllm_server_url http://127.0.0.1:8000/v1 \
  --vllm_server_scoring_mode prompt_logprobs \
  --vllm_server_score_max_tokens 256 \
  --num_workers 40 \
  --generation_max_total_tokens 4096 \
  --generation_max_new_tokens 64 \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 1e-4 \
  --validation_freq 100 \
  --validation_batch_size 4 \
  --validation_max_batches 20 \
  --save_freq 500
```

Dataset presets:

| Dataset | `DATASET` | `RUNNING_NAME` |
| --- | --- | --- |
| HotpotQA | `hotpotqa` | `train_hotpotqa_vllm_server_from_scratch` |
| 2WikiMultiHopQA | `2wikimultihopqa` | `train_2wiki_vllm_server_from_scratch` |
| MuSiQue | `musique` | `train_musique_vllm_server_from_scratch` |

Choose one row, set `DATASET` and `RUNNING_NAME`, then run the training template above.

In this setup, `cuda:0` refers to logical GPU `0` inside the training process. Because `CUDA_VISIBLE_DEVICES=1`, it maps to physical GPU `1`. The Llama model is not loaded by `train.py`; it is served by the vLLM process already running on physical GPUs `5,6`.

Training logs:

```text
logs/train/<dataset>/<running_name>__<timestamp>/train.log
```

Checkpoints:

```text
predictions/<dataset>/<run_name>/...
```

## Inference

Main entry point:

```bash
python -m source.run.inference
```

Use a trained retriever checkpoint from `predictions/<dataset>/...`:

```bash
DATASET=musique
CHECKPOINT_PATH=./predictions/musique/<run_name>/<checkpoint_dir>

CUDA_VISIBLE_DEVICES=5,6 python -m source.run.inference \
  --method rescore \
  --running_name "infer_${DATASET}_best" \
  --dataset "${DATASET}" \
  --dataset_split test \
  --prompt_set 1 \
  --retrieval_query_model_name_or_path "${CHECKPOINT_PATH}" \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0
```

Inference logs:

```text
logs/inference/<dataset>/<running_name>__<timestamp>/inference.log
```

Prediction and evaluation files are written under the corresponding `predictions/...` directory.

## Evaluation

Normal evaluation is run at the end of inference and saved next to predictions.

If the official evaluator for a dataset is missing, the code falls back to internal evaluation instead of crashing. The official evaluation JSON will include a flag such as:

```json
{
  "official_evaluation_skipped": true
}
```

## Important Runtime Notes

When using `generation_backend=vllm_server`, `train.py` does not load Llama locally and does not override server-side vLLM settings such as `gpu_memory_utilization`, `tensor_parallel_size`, or `max_model_len`. Those are controlled only by `script/preload_vllm_server.py`.

The vLLM server client still controls request-side limits:

- `--vllm_server_score_max_tokens`: truncates prompts used for scoring
- `--vllm_server_prompt_logprobs`: controls top-k prompt logprobs requested from vLLM
- `--vllm_server_missing_logprob_fallback`: fallback logprob if target token is outside returned top-k

For RTX 2080 Ti or other Turing GPUs, use:

```text
--dtype half
```

`bfloat16` is not supported on these GPUs.

## Troubleshooting

If vLLM returns `400 Bad Request` with a message like maximum context length exceeded, lower one of:

```bash
--generation_max_total_tokens
--generation_max_new_tokens
```

If vLLM returns `500 Internal Server Error` during scoring, check `train.log` for:

```text
[vllm-server-error]
```

Then reduce request-side scoring load:

```bash
--vllm_server_score_max_tokens 256
--vllm_server_prompt_logprobs 1
```

If the retriever GPU is underused, increase:

```bash
--retrieval_batch_size 256
```

Then try:

```bash
--retrieval_batch_size 512
```

Increase `--retrieval_buffer_size` only if the vLLM server is stable, because it increases the number of LLM scoring prompts per train step.

If DataLoader fails with `Too many open files`, reduce:

```bash
--num_workers 4
```

If the server crashes, restart only the vLLM server. The training code does not kill or replace existing vLLM processes.

## Citation

```bibtex
@inproceedings{lee-etal-2025-rescore,
  title = "{R}e{SCORE}: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision",
  author = "Lee, Dosung and Oh, Wonjun and Kim, Boyoung and Kim, Minyoung and Park, Joonsuk and Seo, Paul Hongsuck",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025"
}
```
