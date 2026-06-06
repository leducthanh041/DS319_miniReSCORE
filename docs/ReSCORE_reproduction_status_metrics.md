# Trạng Thái Reproduction ReSCORE Và Checklist Báo Cáo

Ngày cập nhật: 2026-06-06

Tài liệu này tổng hợp trạng thái reproduction hiện tại của project ReSCORE cục bộ, các phần đã làm, các phần chưa làm, metric cần báo cáo và ablation nên thực hiện. Nội dung dựa trên:

- `docs/ReSCORE-paper.pdf`
- `docs/ReSCORE_Pipeline_for_Reproduction.md`
- execution path thực tế trong `source/`
- các thay đổi/debug đã thực hiện trong repo

## Trạng Thái Cục Bộ Hiện Tại

Repo đang làm việc:

```bash
/mmlab_students/storageStudents/nguyenvd/Thanhld/DS319/ReSCORE
```

Thư mục `predictions` trong repo đã được đổi thành symlink sang Docker local storage:

```bash
predictions -> /docker/data/thanhld/ReSCORE/predictions
```

Mục đích là tránh ghi checkpoint/output nặng trực tiếp xuống filesystem NFS `/mmlab_students`.

Thư mục `predictions` cũ trên NFS đã được giữ lại tại:

```bash
/mmlab_students/storageStudents/nguyenvd/Thanhld/DS319/ReSCORE/predictions_nfs_backup_20260606_112045
```

Symlink thừa `predictions_docker_link` đã được xoá.

## Những Việc Đã Làm

Các phần đã hoàn thành trong project:

- Bổ sung logging cho build/train/inference để stdout/stderr được ghi ra file log.
- Chỉnh script download và preprocessing để dễ debug hơn.
- Rà soát và sửa pipeline build retrieval database cho các dataset multi-hop.
- Thêm progress tracking cho quá trình generate passage embeddings.
- Thêm cơ chế shard embedding theo GPU để giảm rủi ro OOM.
- Thêm resume/skip cho build/index khi embedding đã tồn tại.
- Thiết lập chiến lược hot-data local: log, SQLite/index và checkpoint được ưu tiên lưu trên Docker local storage.
- Thêm một số cấu hình SQLite PRAGMA để giảm áp lực I/O.
- Patch training để hỗ trợ runtime log, safe CUDA cache handling, early stopping, validation limit và `--max_train_steps`.
- Patch training để hỗ trợ external vLLM server.
- Patch vLLM scoring để dùng `prompt_logprobs` và fallback khi logprob không đầy đủ.
- Patch Llama generation để dùng `attention_mask` và tránh resize token embedding gây OOM.
- Patch inference để dùng vLLM với `half` dtype trên RTX 2080 Ti và không crash khi thiếu official evaluator.
- Dọn checkpoint MuSiQue cũ, chỉ giữ lại checkpoint thực tế đang chọn.

## Checkpoint MuSiQue Đang Chọn

Checkpoint hiện tại dùng để inference:

```bash
predictions/musique/train_musique_bs32_resume_step200_safe_oom___llama_3.1_8b_instruct___epoch_0_step_200/multi_retrieval___train/prompt_set__1/retr_count__4/epoch_0_step_200
```

Checkpoint này được chọn vì log cho thấy validation loss tốt nhất:

```text
avg_loss = 0.000004
epoch = 0
step = 200
log = /docker/data/thanhld/ReSCORE/logs/train/musique/train_musique_bs32_resume_step200_safe_oom__20260604_120952/train.log
```

Lưu ý: thư mục `best_validation` trong run này không nhất thiết là checkpoint có validation loss nhỏ nhất tuyệt đối, vì logic early stopping dùng `early_stopping_min_delta=1e-4`. Do đó checkpoint `epoch_0_step_200` đang được ưu tiên thủ công.

## Những Gì Đã Reproduce Được

Hiện tại reproduction đã bao phủ các phần sau:

- Preprocessing và sử dụng retrieval DB cho MuSiQue.
- Train dense retriever Contriever theo hướng ReSCORE-style supervision.
- Dùng Llama-3.1-8B-Instruct qua external vLLM server để generation/scoring trong training.
- Train trong điều kiện VRAM hạn chế bằng cách giảm prompt/context/thought settings so với paper.
- Resume training từ checkpoint sau OOM/reboot.
- Chọn checkpoint MuSiQue thực tế để inference.
- Chuẩn bị inference/evaluation qua `source/run/inference.py`.

## Những Gì Chưa Reproduce Đầy Đủ

Các phần chưa khớp đầy đủ với paper ReSCORE:

- Chưa reproduce đầy đủ cả ba dataset: MuSiQue, HotpotQA, 2WikiMHQA.
- Chưa chạy full paper hyperparameter trên GPU A100-class.
- Chưa giữ được training top `M = 32` document distribution trong điều kiện ổn định VRAM.
- Chưa giữ được inference top `k = 8` nếu dùng config giảm để tránh OOM.
- Chưa giữ đầy đủ `max iterations = 6` và `min iterations = 2` nếu local dùng `max_num_thought` thấp hơn.
- Official evaluator cho toàn bộ dataset có thể chưa đầy đủ hoặc chưa chạy xong.
- Metric retrieval `MHR_i@8` chưa được tính/báo cáo đầy đủ từ inference output hiện tại.
- Chưa chạy các ablation chính của paper.
- Chưa reproduce statistical significance test trên 10 random seeds.

## Metric Cần Báo Cáo Theo Paper

Paper ReSCORE dùng hai nhóm metric chính.

### Metric QA

Với mỗi dataset, cần báo cáo answer-level metrics theo official evaluation protocol:

```text
Answer EM
Answer F1
```

Đây là metric chính trong các bảng Table 1, Table 2, Table 4 và Table 5 của paper.

Với MuSiQue hiện tại, tối thiểu nên báo cáo:

```text
Dataset
Checkpoint
Answer EM
Answer F1
Count
```

### Metric Retrieval

Paper giới thiệu multi-hop recall at k:

```text
MHR_i@k
```

Paper dùng:

```text
k = 8
```

Nên báo cáo:

```text
MHR_1@8
MHR_2@8
MHR_final@8
```

Trong đó `MHR_i@8` là recall tích luỹ của gold supporting documents được retrieve đến iteration `i`. `MHR_final@8` tương ứng với iteration cuối `eta_n`, có thể khác nhau theo từng câu hỏi.

### Metric Phụ Trong Code

Code hiện tại cũng hỗ trợ một số metric nội bộ hữu ích cho debug:

```text
sp_em
sp_f1
sp_precision
sp_recall
answer_support_recall
avg_predicted_paras
```

Các metric này hữu ích để phân tích, nhưng metric retrieval cốt lõi của paper vẫn là `MHR_i@8`.

## Bảng Kết Quả Chính Nên Có

Với báo cáo reproduction tối thiểu:

```text
Dataset | Checkpoint | EM | F1 | MHR_1@8 | MHR_2@8 | MHR_final@8
```

Với checkpoint local hiện tại:

```text
MuSiQue | epoch_0_step_200 | TBD | TBD | TBD | TBD | TBD
```

Nếu tiếp tục reproduce HotpotQA và 2WikiMHQA, bổ sung thêm:

```text
HotpotQA
2WikiMHQA
```

## Ablation Nên Làm Khi Reproduce ReSCORE

Paper có một số ablation quan trọng. Nếu tài nguyên hạn chế, nên ưu tiên theo thứ tự dưới đây.

### 1. Ablation Pseudo-GT Label

So sánh các cách tạo document scoring label:

```text
P_LM(q | d_j)
P_LM(a | q, d_j)
P_LM(q, a | d_j)
```

Ý nghĩa:

- `P_LM(q | d_j)`: đo relevance giữa question và document.
- `P_LM(a | q, d_j)`: đo consistency giữa document và answer.
- `P_LM(q, a | d_j)`: kết hợp relevance và consistency, là pseudo-GT label chính của ReSCORE.

Metric cần báo cáo:

```text
Recall@k sau reranking
EM
F1
```

Ablation này tương ứng với Table 3 trong paper.

### 2. Ablation Pseudo-GT So Với GT Label

So sánh các kiểu label để train retriever:

```text
None
GT
Pseudo-GT
```

Ý nghĩa:

- `None`: không fine-tune retriever.
- `GT`: train bằng ground-truth supporting document labels.
- `Pseudo-GT`: train bằng pseudo labels của ReSCORE.

Metric cần báo cáo:

```text
EM
F1
MHR_1@8
MHR_2@8
MHR_final@8
```

Ablation này tương ứng với Table 4 trong paper.

### 3. Ablation Query Reformulation

So sánh các cách tạo query cho iteration tiếp theo:

```text
None
LLM-rewrite
Thought-concat
```

Ý nghĩa:

- `None`: luôn dùng original question ở mọi retrieval step.
- `LLM-rewrite`: dùng LLM rewrite query dựa trên retrieved evidence.
- `Thought-concat`: nối thought sinh ra với query hiện tại/original query.

Metric cần báo cáo:

```text
EM
F1
MHR_1@8
MHR_2@8
MHR_final@8
```

Ablation này tương ứng với Table 5 trong paper.

### 4. Optional: Kiểm Định Nhiều Random Seed

Paper có báo cáo significance testing trên 10 random seeds ở Appendix C.

Nếu làm reproduction chính thức, có thể chạy nhiều seed và báo cáo:

```text
mean EM/F1
standard deviation
p-value hoặc confidence interval
```

Nếu tài nguyên tính toán hạn chế, có thể ghi rõ phần này chưa reproduce.

## Hyperparameter Paper Để Đối Chiếu

Paper mô tả các thiết lập chính:

```text
LLM: Llama-3.1-8B-Instruct
Retriever: Contriever
Training: train query/question embedder, freeze document embedder
Training top M documents: 32
Inference top k documents: 8
Maximum iterations: 6
Minimum iterations: 2
Batch size: 16
Retriever distribution temperature: 0.1
Optimizer: AdamW
Initial learning rate: 1e-6
LR decay: exponential decay 0.9 mỗi 100 iterations
Early stopping: dừng khi validation loss không cải thiện trong một epoch
Hardware paper: 2 x NVIDIA A100 40GB
```

## Khác Biệt Cục Bộ So Với Paper

Reproduction hiện tại khác paper vì giới hạn GPU/VRAM và độ ổn định server:

- Dùng RTX-class GPU VRAM thấp thay vì A100 40GB.
- Dùng external vLLM server để giữ Llama tách khỏi process train retriever.
- Dùng nhiều OOM-safe configs trong training và inference.
- Giảm `generation_max_total_tokens`, `generation_max_model_len`, `retrieval_buffer_size`, `prompt_max_para_count`, `prompt_max_para_words` và `max_num_thought`.
- Checkpoint MuSiQue hiện tại đến từ chuỗi resumed training, không phải một full clean paper run.
- Chọn checkpoint theo validation loss local, chưa phải full official test-set comparison.
- Checkpoint/output được symlink sang Docker local storage để tránh NFS I/O hang.

Các điểm này nên được ghi rõ trong báo cáo reproduction.

## Bước Tiếp Theo Đề Xuất

1. Chạy inference MuSiQue với checkpoint đã chọn.
2. Lưu normal evaluation và official evaluation JSON nếu official evaluator hoạt động.
3. Báo cáo `EM` và `F1` cho MuSiQue.
4. Kiểm tra hoặc bổ sung tính `MHR_i@8` từ retrieval traces.
5. Nếu tài nguyên cho phép, chạy cùng workflow cho HotpotQA và 2WikiMHQA.
6. Nếu cần chứng minh claim cốt lõi của ReSCORE, ưu tiên ablation pseudo-GT label.
7. Nếu cần chứng minh thiết kế IQATR, chạy thêm ablation query reformulation.

## Lệnh Inference Template Cho Checkpoint Hiện Tại

```bash
CUDA_VISIBLE_DEVICES=5,6 python -m source.run.inference \
    --method rescore \
    --running_name infer_musique_best_step200_vllm \
    --dataset musique \
    --dataset_split test \
    --prompt_set 1 \
    --batch_size 8 \
    --retrieval_query_model_name_or_path "predictions/musique/train_musique_bs32_resume_step200_safe_oom___llama_3.1_8b_instruct___epoch_0_step_200/multi_retrieval___train/prompt_set__1/retr_count__4/epoch_0_step_200" \
    --retrieval_passage_model_name_or_path facebook/contriever-msmarco \
    --retriever_device cuda:0 \
    --retrieval_count 4 \
    --retrieval_buffer_size 4 \
    --retrieval_batch_size 64 \
    --max_num_thought 2 \
    --prompt_max_para_count 8 \
    --prompt_max_para_words 220 \
    --generation_dtype half \
    --generation_tensor_parallel_size 2 \
    --generation_gpu_memory_utilization 0.9 \
    --generation_max_model_len 2048 \
    --generation_max_total_tokens 2048 \
    --generation_max_batch_size 1 \
    --generation_max_new_tokens 48 \
    --generation_swap_space 0 \
    --generation_cpu_offload_gb 0 \
    --runtime_log_root /docker/data/thanhld/ReSCORE/logs/inference
```

Nếu dùng lệnh này để báo cáo kết quả, cần ghi rõ đây là config đã giảm so với paper vì giới hạn phần cứng.
