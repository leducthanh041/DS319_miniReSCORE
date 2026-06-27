# ReSCORE-TTA Workspace

This repository is an ongoing research codebase for multi-hop question answering with iterative retrieval, ReSCORE-style retriever training, and test-time adaptation (TTA). The project is not yet published.

Active code lives in `source/`.

## 1. Installation

```bash
conda create -n rescore python=3.10 -y
conda activate rescore
pip install -r requirements.txt

export HF_TOKEN=<your_huggingface_token>
huggingface-cli login --token "$HF_TOKEN"
```

## 2. Data And Local Storage

Use local disk for logs, checkpoints, SQLite/FAISS files when running on shared NFS:

```bash
bash script/setup_local_hot_data.sh
```

Prepare data and retrieval DB:

```bash
bash script/download/multihop_raw_data.sh
bash script/download/multihop_processed_data.sh
bash script/download/build.sh
```

Supported datasets:

```text
musique
hotpotqa
2wikimultihopqa
```

## 3. Start vLLM Server

Run the LLM on separate GPUs, for example physical GPUs `5,6`:

```bash
python script/preload_vllm_server.py \
  --cuda_visible_devices 5,6 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor_parallel_size 2 \
  --dtype half \
  --gpu_memory_utilization 0.90 \
  --max_model_len 4096 \
  --max_num_seqs 20 \
  --swap_space 0 \
  --cpu_offload_gb 0 \
  --enforce_eager \
  --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/models
```

## 4. Common Variables

Change only these variables for most runs:

```bash
export DATASET=musique
export RETRIEVER_GPU=1
export VLLM_URL=http://127.0.0.1:8000/v1
export DB=./data/database/contriever_msmarco/${DATASET}
export CKPT=./predictions/musique/<train_run>/multi_retrieval___train/prompt_set__1/retr_count__8/best_validation
```

For OOD evaluation, keep `CKPT` fixed and change `DATASET`/`DB`:

```bash
export DATASET=hotpotqa
export DB=./data/database/contriever_msmarco/${DATASET}
```

## 5. Train

```bash
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU python -m source.run.train \
  --running_name train_${DATASET}_baseline \
  --dataset "$DATASET" \
  --method rescore \
  --prompt_set 1 \
  --batch_size 16 \
  --n_epochs 1 \
  --lr 1e-6 \
  --retrieval_query_model_name_or_path facebook/contriever-msmarco \
  --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
  --retriever_device cuda:0 \
  --generation_backend vllm_server \
  --vllm_server_url "$VLLM_URL" \
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
  --validation_freq 100 \
  --validation_batch_size 2 \
  --validation_max_batches 20 \
  --save_freq 10
```

## 6. Baseline Inference

```bash
COMMON_INFER_ARGS=(
  --dataset_split test
  --prompt_set 1
  --batch_size 1
  --retrieval_query_model_name_or_path "$CKPT"
  --database_path "$DB"
  --retriever_device cuda:0
  --generation_backend vllm_server
  --vllm_server_url "$VLLM_URL"
  --vllm_server_max_model_len 2048
  --vllm_server_context_safety_margin 16
  --generation_max_batch_size 1
  --generation_max_total_tokens 4096
  --generation_max_new_tokens 48
  --retrieval_count 8
  --retrieval_buffer_size 32
  --retrieval_batch_size 32
  --retrieval_no_duplicates
  --max_num_thought 6
  --prompt_max_para_count 8
  --prompt_max_para_words 200
)
```

```bash
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU python -m source.run.inference \
  --running_name infer_${DATASET}_baseline \
  --dataset "$DATASET" \
  --method rescore \
  "${COMMON_INFER_ARGS[@]}"
```

## 7. TTA Inference

```bash
COMMON_TTA_ARGS=(
  "${COMMON_INFER_ARGS[@]}"
  --generation_model_name meta-llama/Llama-3.1-8B-Instruct
  --vllm_server_scoring_mode prompt_logprobs
  --vllm_server_score_max_tokens 512
  --vllm_server_prompt_logprobs 20
  --vllm_server_missing_logprob_fallback -20.0
  --tta_level both
  --tta_pseudo_label dual
  --tta_inner_steps 3
  --tta_query_lr 1.2
  --tta_momentum 0.99
  --tta_weight_decay 0.01
  --tta_temperature 0.5
  --tta_nucleus_p 0.5
  --tta_anchor_weight 0.1
  --tta_max_grad_norm 1.0
  --tta_warmup_steps 0
  --tta_refresh_candidates_each_step
  --tta_confidence_threshold 0.0
  --tta_fail_on_pseudo_label_error
  --tta_lora_rank 8
  --tta_lora_alpha 16
  --tta_lora_lr 5e-4
  --tta_lora_loss_weight 1.0
  --tta_lora_num_top_layers 4
  --tta_lora_reg_weight 0.01
  --tta_cross_encoder_device cuda:0
  --tta_cross_encoder_batch_size 32
  --tta_cross_encoder_max_length 512
  --tta_clear_cross_encoder_cache
  --tta_log_every 1
)
```

Soft TTA:

```bash
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU python -m source.run.inference_tta \
  --running_name infer_tta_${DATASET}_soft \
  --dataset "$DATASET" \
  --method iqatr_tta \
  "${COMMON_TTA_ARGS[@]}"
```

Hard TTA:

```bash
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU python -m source.run.inference_tta_hard \
  --running_name infer_tta_${DATASET}_hard \
  --dataset "$DATASET" \
  --method iqatr_tta_hard \
  "${COMMON_TTA_ARGS[@]}"
```

## Reference

This project references the ReSCORE paper and implementation for the original iterative retriever training idea:

- Paper: <https://arxiv.org/abs/2505.21250>
- Code: <https://github.com/leeds1219/ReSCORE>
- Project page: <https://leeds1219.github.io/ReSCORE>
