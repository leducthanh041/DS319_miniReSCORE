# Các Bảng Thí Nghiệm Cần Báo Cáo Cho ReSCORE

Ngày cập nhật: 2026-06-06

Tài liệu này tổng hợp các bảng nên đưa vào báo cáo reproduction ReSCORE. Các bảng chỉ giữ những hạng mục paper ReSCORE có nêu rõ hoặc có sử dụng trong phần experiment/ablation. Những thông tin paper không nêu sẽ không được liệt kê trong bảng.

Nguồn đối chiếu:

- `docs/ReSCORE-paper.pdf`
- `docs/ReSCORE_reproduction_status_metrics.md`
- `docs/ReSCORE_paper_vs_local_hyperparameters.md`

## Bảng 1. Setup Thí Nghiệm Chính

| Hạng mục | Paper ReSCORE | Reproduction cục bộ | Ghi chú |
|---|---|---|---|
| Dataset | MuSiQue, HotpotQA, 2WikiMHQA | MuSiQue | Hiện tại mới chọn checkpoint MuSiQue để inference. |
| Method | ReSCORE / IQATR | ReSCORE local reproduction | Giữ ý tưởng train retriever bằng pseudo-GT relevance-consistency. |
| LLM | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | Giữ cùng model LLM. |
| Retriever | Contriever | Contriever | Giữ cùng retriever family. |
| Query encoder | Train question/query embedder | Train query encoder | Giữ cùng hướng train. |
| Document encoder | Frozen | Frozen / passage model không train | Giữ cùng ý tưởng với paper. |
| Hardware | 2 x NVIDIA A100 40GB | RTX 2080 Ti-class GPU khoảng 10.75GiB VRAM | Khác biệt chính, cần nêu trong báo cáo. |

## Bảng 2. Checkpoint Được Chọn Để Inference

| Dataset | Running name | Checkpoint | Tiêu chí chọn | Validation loss | Log nguồn |
|---|---|---|---|---:|---|
| MuSiQue | `train_musique_bs32_resume_step200_safe_oom` | `predictions/musique/train_musique_bs32_resume_step200_safe_oom___llama_3.1_8b_instruct___epoch_0_step_200/multi_retrieval___train/prompt_set__1/retr_count__4/epoch_0_step_200` | Validation loss thấp nhất quan sát được | `0.000004` | `/docker/data/thanhld/ReSCORE/logs/train/musique/train_musique_bs32_resume_step200_safe_oom__20260604_120952/train.log` |

Ghi chú: checkpoint này được chọn thủ công theo validation loss thực tế. Thư mục `best_validation` trong run này không được ưu tiên vì logic `early_stopping_min_delta=1e-4`.

## Bảng 3. Hyperparameter Paper So Với Cục Bộ

| Nhóm | Hyperparameter | Paper ReSCORE | Reproduction cục bộ | Lý do thay đổi |
|---|---|---:|---:|---|
| Training | Batch size | `16` | `32` | Tăng throughput, bù lại giảm retrieval/context để tránh OOM. |
| Training | Learning rate | `1e-6` | `2e-5` | Config local hiện tại dùng LR cao hơn paper. |
| Training | Optimizer | AdamW | AdamW | Giữ nguyên. |
| Training | LR decay | Exponential decay `0.9` mỗi 100 iterations | Có decay trong code | Giữ cùng ý tưởng schedule. |
| Training | Retriever temperature | `0.1` | `0.1` | Giữ nguyên. |
| Training | Early stopping | Dừng khi validation loss không cải thiện trong một epoch | Bật early stopping | Giữ cùng ý tưởng, nhưng local validation bị giới hạn. |
| Retrieval | Training top `M` documents | `32` | `retrieval_buffer_size=4` | Giảm mạnh để tránh OOM và giảm chi phí LLM scoring. |
| Retrieval | Inference top `k` documents | `8` | `retrieval_count=4` | Giảm để phù hợp VRAM/context local. |
| Iteration | Maximum iterations | `6` | `max_num_thought=2` | Giảm runtime, prompt length và OOM risk. |
| Iteration | Minimum iterations | `2` | Chưa đảm bảo giống paper | Cần ghi là khác biệt implementation nếu code không enforce. |
| Hardware | GPU memory | 40GB/GPU | khoảng 10.75GiB/GPU | Lý do chính dẫn đến giảm hyperparameter. |

## Bảng 4. Kết Quả QA Chính Cần Báo Cáo

Paper báo cáo answer-level `EM` và `F1` theo official evaluation protocol của từng dataset.

| Dataset | Method | Checkpoint | EM | F1 | Count | Trạng thái |
|---|---|---|---:|---:|---:|---|
| MuSiQue | ReSCORE local reproduction | `epoch_0_step_200` | TBD | TBD | TBD | Cần hoàn tất inference/evaluation. |
| HotpotQA | ReSCORE local reproduction | TBD | TBD | TBD | TBD | Chưa reproduce đầy đủ. |
| 2WikiMHQA | ReSCORE local reproduction | TBD | TBD | TBD | TBD | Chưa reproduce đầy đủ. |

## Bảng 5. Kết Quả Retrieval Cần Báo Cáo

Paper dùng metric multi-hop recall:

```text
MHR_i@8
```

Nên báo cáo tại `i=1`, `i=2` và iteration cuối `eta_n`.

| Dataset | Method | Checkpoint | MHR_1@8 | MHR_2@8 | MHR_final@8 | Trạng thái |
|---|---|---|---:|---:|---:|---|
| MuSiQue | ReSCORE local reproduction | `epoch_0_step_200` | TBD | TBD | TBD | Cần retrieval trace/support labels. |
| HotpotQA | ReSCORE local reproduction | TBD | TBD | TBD | TBD | Chưa reproduce đầy đủ. |
| 2WikiMHQA | ReSCORE local reproduction | TBD | TBD | TBD | TBD | Chưa reproduce đầy đủ. |

Nếu chưa kịp tính `MHR_i@8`, trong báo cáo cần ghi rõ:

```text
EM/F1 đã được báo cáo trước; MHR_i@8 sẽ được bổ sung khi hoàn tất phân tích retrieval trace.
```

## Bảng 6. Ablation Pseudo-GT Label

Ablation này tương ứng với Table 3 trong paper. Mục tiêu là kiểm tra tác dụng của từng loại pseudo-GT label.

| Dataset | Label scoring | Recall@k / R@k | EM | F1 | Trạng thái |
|---|---|---:|---:|---:|---|
| MuSiQue | `P_LM(q | d_j)` | TBD | TBD | TBD | Chưa chạy. |
| MuSiQue | `P_LM(a | q, d_j)` | TBD | TBD | TBD | Chưa chạy. |
| MuSiQue | `P_LM(q, a | d_j)` | TBD | TBD | TBD | Chưa chạy; đây là label chính của ReSCORE. |
| HotpotQA | `P_LM(q | d_j)` | TBD | TBD | TBD | Chưa chạy. |
| HotpotQA | `P_LM(a | q, d_j)` | TBD | TBD | TBD | Chưa chạy. |
| HotpotQA | `P_LM(q, a | d_j)` | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | `P_LM(q | d_j)` | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | `P_LM(a | q, d_j)` | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | `P_LM(q, a | d_j)` | TBD | TBD | TBD | Chưa chạy. |

Ý nghĩa:

- `P_LM(q | d_j)`: relevance giữa question và document.
- `P_LM(a | q, d_j)`: consistency giữa document và answer.
- `P_LM(q, a | d_j)`: kết hợp relevance và consistency, là pseudo-GT label chính của ReSCORE.

## Bảng 7. Ablation Pseudo-GT So Với GT Label

Ablation này tương ứng với Table 4 trong paper. Mục tiêu là so sánh ReSCORE pseudo-GT với việc dùng ground-truth supporting document labels.

| Dataset | Label train retriever | EM | F1 | MHR_1@8 | MHR_2@8 | MHR_final@8 | Trạng thái |
|---|---|---:|---:|---:|---:|---:|---|
| MuSiQue | None | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| MuSiQue | GT | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| MuSiQue | Pseudo-GT / ReSCORE | TBD | TBD | TBD | TBD | TBD | Đang có checkpoint local, cần inference/eval. |
| HotpotQA | None | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| HotpotQA | GT | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| HotpotQA | Pseudo-GT / ReSCORE | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | None | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | GT | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | Pseudo-GT / ReSCORE | TBD | TBD | TBD | TBD | TBD | Chưa chạy. |

## Bảng 8. Ablation Query Reformulation

Ablation này tương ứng với Table 5 trong paper. Mục tiêu là đánh giá cách tạo query cho iteration tiếp theo.

| Dataset | Query reformulation | EM | F1 | Trạng thái |
|---|---|---:|---:|---|
| MuSiQue | None | TBD | TBD | Chưa chạy. |
| MuSiQue | LLM-rewrite | TBD | TBD | Chưa chạy. |
| MuSiQue | Thought-concat | TBD | TBD | Pipeline local gần hướng này, cần xác nhận/tách ablation. |
| HotpotQA | None | TBD | TBD | Chưa chạy. |
| HotpotQA | LLM-rewrite | TBD | TBD | Chưa chạy. |
| HotpotQA | Thought-concat | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | None | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | LLM-rewrite | TBD | TBD | Chưa chạy. |
| 2WikiMHQA | Thought-concat | TBD | TBD | Chưa chạy. |

Paper dùng bảng này để phân tích `None`, `LLM-rewrite` và `Thought-concat`.

## Bảng 9. Trạng Thái Các Thí Nghiệm

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Train ReSCORE MuSiQue | Đã chạy một phần | Có checkpoint local `epoch_0_step_200`. |
| Inference MuSiQue | Đang/chưa hoàn tất | Cần kiểm tra log inference và file prediction/evaluation. |
| MuSiQue EM/F1 | Chưa có kết quả cuối | Cần inference/evaluation hoàn tất. |
| MuSiQue MHR_i@8 | Chưa có kết quả cuối | Cần retrieval trace/support labels. |
| HotpotQA ReSCORE | Chưa hoàn tất | Cần train/inference/eval. |
| 2WikiMHQA ReSCORE | Chưa hoàn tất | Cần train/inference/eval. |
| Pseudo-GT label ablation | Chưa chạy | Quan trọng nhất nếu cần chứng minh claim ReSCORE. |
| GT vs Pseudo-GT ablation | Chưa chạy | Cần nếu so sánh với supervised document labels. |
| Query reformulation ablation | Chưa chạy | Cần nếu phân tích thiết kế IQATR. |

## Bảng 10. Bảng Báo Cáo Tối Thiểu Nếu Chỉ Có MuSiQue

Nếu thời gian hạn chế, có thể báo cáo tối thiểu:

| Dataset | Method | Hardware | Checkpoint | EM | F1 | Ghi chú |
|---|---|---|---|---:|---:|---|
| MuSiQue | ReSCORE local reproduction | RTX 2080 Ti-class + vLLM server | `epoch_0_step_200` | TBD | TBD | Config đã giảm so với paper do giới hạn VRAM. |

Kèm theo câu mô tả:

```text
Do giới hạn phần cứng, reproduction này không dùng đúng setup A100 40GB như paper. Chúng tôi giữ ý tưởng chính của ReSCORE nhưng giảm training top M, inference top k, độ dài context và số reasoning iteration tối đa để chạy ổn định trên GPU khoảng 10.75GiB VRAM.
```

