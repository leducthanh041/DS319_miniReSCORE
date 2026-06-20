# ReSCORE Reproduction Workspace

This repository is a practical reproduction workspace for **ReSCORE**, inspired by the original project: <https://github.com/leeds1219/ReSCORE>.

Paper: <https://arxiv.org/abs/2505.21250>  
Project page: <https://leeds1219.github.io/ReSCORE>

The active execution path in this repo is under `source/`. Notes in `docs/` explain the paper/repo differences, but actual training and inference should be checked against `source/`.

## Overview

ReSCORE trains a dense retriever for multi-hop question answering using LLM-based relevance-consistency supervision. This workspace supports:

- datasets: `hotpotqa`, `2wikimultihopqa`, `musique`
- retriever: `facebook/contriever-msmarco` or a trained checkpoint
- generator: `meta-llama/Llama-3.1-8B-Instruct`
- index: FAISS + SQLite docstore
- training/inference logs under `logs/...`
- checkpoints and predictions under `predictions/...`

Main entry points:

```text
source/run/train.py
source/run/inference.py
script/preload_vllm_server.py
script/download/
```

## Installation

```bash
pip install -r requirements.txt
```

For Llama 3.1, authenticate Hugging Face first:

```bash
export HF_TOKEN=<your_huggingface_token>
```

If you use a fine-grained token, enable access to public gated repositories.

## Storage

On shared NFS, heavy writes from logs, SQLite, FAISS, and checkpoints can stall I/O. Prefer local SSD/Docker storage for hot data:

```bash
mkdir -p /docker/data/$USER/ReSCORE/logs
mkdir -p /docker/data/$USER/ReSCORE/data/database
mkdir -p /docker/data/$USER/ReSCORE/predictions

ln -s /docker/data/$USER/ReSCORE/logs logs
ln -s /docker/data/$USER/ReSCORE/data/database data/database
ln -s /docker/data/$USER/ReSCORE/predictions predictions
```

If your machine has enough local disk and stable I/O, these paths can also stay inside the repo.

## Data Preparation

Download raw data:

```bash
bash script/download/multihop_raw_data.sh
```

Create processed data:

```bash
bash script/download/multihop_processed_data.sh
```

Build retrieval DB and FAISS index:

```bash
bash script/download/build.sh
```

Expected files:

```text
data/processed_data/<dataset>/{train,dev_subsampled,test_subsampled}.jsonl
data/database/contriever_msmarco/<dataset>/{docstore.db,index.faiss,faiss_id_to_docstore_id.pkl}
```

## vLLM Server

For stable training/inference on limited VRAM, run Llama in a persistent vLLM server on separate GPUs. Example: GPU `5,6` for vLLM, GPU `1` for the retriever client.

Start vLLM:

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

Check server:

```bash
curl http://127.0.0.1:8000/v1/models
```

This script does not kill existing vLLM processes. Stop them manually only when needed.

## Training

Use the same template for all supported datasets by changing `DATASET` and `RUNNING_NAME`.

```bash
DATASET=musique
RUNNING_NAME=train_${DATASET}_baseline

CUDA_VISIBLE_DEVICES=1 python -m source.run.train \
  --running_name "${RUNNING_NAME}" \
  --dataset "${DATASET}" \
  --method rescore \
  --prompt_set 1 \
  --batch_size 16 \
  --n_epochs 1 \
  --lr 1e-6 \
  --retrieval_query_model_name_or_path facebook/contriever-msmarco \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0 \
  --generation_backend vllm_server \
  --vllm_server_url http://127.0.0.1:8000/v1 \
  --vllm_server_max_model_len 2048 \
  --vllm_server_context_safety_margin 16 \
  --vllm_server_scoring_mode prompt_logprobs \
  --vllm_server_score_max_tokens 96 \
  --vllm_server_prompt_logprobs 1 \
  --vllm_server_missing_logprob_fallback -20.0 \
  --generation_max_batch_size 1 \
  --generation_max_total_tokens 2048 \
  --generation_max_new_tokens 48 \
  --retrieval_count 8 \
  --retrieval_buffer_size 32 \
  --retrieval_batch_size 32 \
  --max_num_thought 6 \
  --prompt_max_para_count 8 \
  --prompt_max_para_words 200 \
  --num_workers 2 \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 1e-4 \
  --validation_freq 100 \
  --validation_batch_size 2 \
  --validation_max_batches 20 \
  --save_freq 10
```

With `CUDA_VISIBLE_DEVICES=1`, `--retriever_device cuda:0` means physical GPU `1`. Llama is not loaded by `train.py`; it is served by the vLLM process.

Checkpoints are saved under:

```text
predictions/<dataset>/<run_name>/multi_retrieval___train/prompt_set__<id>/retr_count__<k>/
```

## Inference

Recommended setup: keep the vLLM server on GPUs `5,6`, then run the inference client on a separate retriever GPU, for example physical GPU `1`.

Use a MuSiQue-trained retriever checkpoint for both in-domain and OOD evaluation:

```bash
CKPT=./predictions/musique/train_musique_baseline_resume_best___llama_3.1_8b_instruct___best_validation/multi_retrieval___train/prompt_set__1/retr_count__8/best_validation
```

### Baseline

```bash
BASE_COMMON_ARGS="\
--method rescore \
--dataset_split test \
--prompt_set 1 \
--batch_size 1 \
--retrieval_query_model_name_or_path ${CKPT} \
--retriever_device cuda:0 \
--generation_backend vllm_server \
--vllm_server_url http://127.0.0.1:8000/v1 \
--vllm_server_max_model_len 2048 \
--vllm_server_context_safety_margin 16 \
--generation_max_batch_size 1 \
--generation_max_total_tokens 4096 \
--generation_max_new_tokens 48 \
--retrieval_count 8 \
--retrieval_buffer_size 32 \
--retrieval_batch_size 32 \
--retrieval_no_duplicates \
--max_num_thought 6 \
--prompt_max_para_count 8 \
--prompt_max_para_words 200 \
--prediction_root /docker/data/$USER/ReSCORE/predictions \
--runtime_log_root /docker/data/$USER/ReSCORE/logs/inference"
```

Run MuSiQue in-domain baseline:

```bash
CUDA_VISIBLE_DEVICES=1 python -m source.run.inference \
  --running_name infer_musique_fair_baseline \
  --dataset musique \
  --database_path ./data/database/contriever_msmarco/musique \
  ${BASE_COMMON_ARGS}
```

### Test-Time Adaptation

TTA uses the same generator, retriever checkpoint, retrieval DB, and prompt budget as the baseline. It additionally performs per-test query adaptation and LoRA adaptation.

```bash
TTA_COMMON_ARGS="\
--method iqatr_tta \
--dataset_split test \
--prompt_set 1 \
--batch_size 1 \
--retrieval_query_model_name_or_path ${CKPT} \
--retriever_device cuda:0 \
--generation_model_name meta-llama/Llama-3.1-8B-Instruct \
--generation_backend vllm_server \
--vllm_server_url http://127.0.0.1:8000/v1 \
--vllm_server_scoring_mode prompt_logprobs \
--vllm_server_score_max_tokens 512 \
--vllm_server_prompt_logprobs 20 \
--vllm_server_max_model_len 2048 \
--vllm_server_missing_logprob_fallback -20.0 \
--vllm_server_context_safety_margin 16 \
--generation_max_batch_size 1 \
--generation_max_total_tokens 4096 \
--generation_max_new_tokens 48 \
--retrieval_count 8 \
--retrieval_buffer_size 32 \
--retrieval_no_duplicates \
--retrieval_batch_size 32 \
--max_num_thought 6 \
--prompt_max_para_count 8 \
--prompt_max_para_words 200 \
--tta_level both \
--tta_pseudo_label dual \
--tta_inner_steps 3 \
--tta_query_lr 1.2 \
--tta_momentum 0.99 \
--tta_weight_decay 0.01 \
--tta_temperature 0.5 \
--tta_nucleus_p 0.5 \
--tta_anchor_weight 0.1 \
--tta_max_grad_norm 1.0 \
--tta_warmup_steps 0 \
--tta_refresh_candidates_each_step \
--tta_confidence_threshold 0.0 \
--tta_fail_on_pseudo_label_error \
--tta_lora_rank 8 \
--tta_lora_alpha 16 \
--tta_lora_lr 5e-4 \
--tta_lora_loss_weight 1.0 \
--tta_lora_num_top_layers 4 \
--tta_lora_reg_weight 0.01 \
--tta_cross_encoder_device cuda:0 \
--tta_cross_encoder_batch_size 32 \
--tta_cross_encoder_max_length 512 \
--tta_clear_cross_encoder_cache \
--tta_log_every 1 \
--runtime_log_root /docker/data/$USER/ReSCORE/logs/inference_tta \
--prediction_root /docker/data/$USER/ReSCORE/predictions"
```

Run MuSiQue in-domain TTA:

```bash
CUDA_VISIBLE_DEVICES=1 python -m source.run.inference_tta \
  --running_name infer_tta_musique_ind_both_dual_full \
  --dataset musique \
  --database_path ./data/database/contriever_msmarco/musique \
  ${TTA_COMMON_ARGS}
```

Run OOD TTA using the same MuSiQue checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 python -m source.run.inference_tta \
  --running_name infer_tta_hotpotqa_ood_musique_ckpt_both_dual_full \
  --dataset hotpotqa \
  --database_path ./data/database/contriever_msmarco/hotpotqa \
  ${TTA_COMMON_ARGS}

CUDA_VISIBLE_DEVICES=1 python -m source.run.inference_tta \
  --running_name infer_tta_2wiki_ood_musique_ckpt_both_dual_full \
  --dataset 2wikimultihopqa \
  --database_path ./data/database/contriever_msmarco/2wikimultihopqa \
  ${TTA_COMMON_ARGS}
```

Use `--max_inference_examples 10` for a smoke test. If vLLM reports context-length errors, increase `--vllm_server_context_safety_margin` to `32`. If vLLM runs out of memory, reduce `--vllm_server_score_max_tokens`, `--prompt_max_para_count`, or `--prompt_max_para_words`.

Outputs are written under:

```text
predictions/<dataset>/<run_name>/multi_retrieval___inference/prompt_set__<id>/best/
  test_prediction.json
  test_evaluation.json
  test_official_evaluation.json
  test_retrieval_trace.jsonl
  test_retrieval_evaluation.json
  test_retrieval_per_question.jsonl
  runtime_arguments.json
  configuration.json
```

TTA additionally writes `test_tta_summary.json`.

## Evaluation

Inference automatically writes normal evaluation results to `test_evaluation.json`.

If an official evaluator is available, `test_official_evaluation.json` is also produced. If the official evaluator is missing, the code skips it safely and writes a warning/result marker instead of crashing.

Retrieval evaluation is written to `test_retrieval_evaluation.json`. The main metric is cumulative multi-hop recall:

```text
MHR_i@k = recall of gold supporting documents retrieved from iteration 1 through i
```

For ReSCORE-style reporting, use `k = retrieval_count`. With `--retrieval_count 8`, report:

```text
MHR_1@8
MHR_2@8
MHR_final@8
```

The full retrieval audit trail is stored in `test_retrieval_trace.jsonl`; per-question recall details are stored in `test_retrieval_per_question.jsonl`.

Main metrics to report for QA:

- `em`
- `f1`
- `precision`
- `recall`
- `count`

## Troubleshooting

- If vLLM reports context length errors, reduce `--generation_max_total_tokens`, `--generation_max_new_tokens`, `--prompt_max_para_count`, or `--prompt_max_para_words`.
- If CUDA/NCCL errors appear with local vLLM, use `--generation_backend vllm_server` and separate the vLLM server GPUs from the retriever GPU.
- If the retriever causes OOM, reduce `--retrieval_batch_size` first.
- If DataLoader reports too many open files, reduce `--num_workers`.
- For RTX 2080 Ti / Turing GPUs, use `--dtype half` when starting vLLM.

## Citation

```bibtex
@inproceedings{lee-etal-2025-rescore,
  title = "{R}e{SCORE}: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision",
  author = "Lee, Dosung and Oh, Wonjun and Kim, Boyoung and Kim, Minyoung and Park, Joonsuk and Seo, Paul Hongsuck",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025"
}
```
