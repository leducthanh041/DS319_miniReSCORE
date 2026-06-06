# So Sánh Hyperparameter Giữa Paper ReSCORE Và Reproduction Cục Bộ

Ngày cập nhật: 2026-06-06

Tài liệu này thống kê các thay đổi giữa cấu hình trong paper ReSCORE và cấu hình reproduction hiện tại trong repo. Mục tiêu hiện tại **không phải reproduce y hệt paper**, mà là chạy được pipeline ReSCORE trong điều kiện phần cứng hạn chế, đồng thời giữ lại ý tưởng cốt lõi: dùng tín hiệu relevance-consistency từ LLM để train retriever.

## Checkpoint Cục Bộ Đang Chọn

Checkpoint MuSiQue hiện tại được chọn để inference:

```bash
predictions/musique/train_musique_bs32_resume_step200_safe_oom___llama_3.1_8b_instruct___epoch_0_step_200/multi_retrieval___train/prompt_set__1/retr_count__4/epoch_0_step_200
```

Checkpoint này được chọn dựa trên log:

```bash
/docker/data/thanhld/ReSCORE/logs/train/musique/train_musique_bs32_resume_step200_safe_oom__20260604_120952/train.log
```

Validation loss tốt nhất quan sát được:

```text
avg_loss = 0.000004
epoch = 0
step = 200
```

## So Sánh Phần Cứng

| Hạng mục | Paper ReSCORE | Reproduction cục bộ | Thay đổi / Lý do |
|---|---:|---:|---|
| GPU train | 2 x NVIDIA A100 | RTX 2080 Ti-class GPU | Phần cứng cục bộ yếu hơn đáng kể. |
| VRAM mỗi GPU | 40GB | khoảng 10.75GiB | Đây là nguyên nhân chính phải giảm context, retrieval depth và generation batch size. |
| Chạy LLM | Chạy trong setup paper với GPU mạnh | Tách Llama qua external vLLM server | Giảm nguy cơ OOM khi train retriever. |
| Process retriever | Paper không mô tả tách riêng | `CUDA_VISIBLE_DEVICES=3`, logical `cuda:0` | Retriever train trên một GPU nhìn thấy. |
| Lưu checkpoint/output | Không nêu rõ | `predictions -> /docker/data/thanhld/ReSCORE/predictions` | Tránh nghẽn I/O trên NFS. |
| Lưu log | Không nêu rõ | `/docker/data/thanhld/ReSCORE/logs/...` | Giảm ghi liên tục xuống `/mmlab_students`. |

## Hyperparameter Train

| Hyperparameter | Paper ReSCORE | Reproduction cục bộ | Thay đổi / Lý do |
|---|---:|---:|---|
| Dataset | MuSiQue, HotpotQA, 2WikiMHQA | MuSiQue | Hiện tại mới chọn checkpoint MuSiQue để inference. |
| Method | ReSCORE / IQATR | `rescore`, `multi_retrieval` | Giữ cùng ý tưởng chính. |
| LLM | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | Giữ nguyên model. |
| Retriever | Contriever | Contriever | Giữ nguyên họ retriever. |
| Encoder được train | Question/query embedder | Query encoder | Giữ đúng hướng train retriever. |
| Document encoder | Frozen | Frozen / passage model không train | Giữ cùng ý tưởng với paper. |
| Batch size | 16 | 32 | Tăng batch size để tăng throughput, các phần khác được giảm để tránh OOM. |
| Số epoch | Train đến khi validation loss không cải thiện trong một epoch | `n_epochs=1`, có resume nhiều lần | Do OOM/reboot nên train theo checkpoint/resume. |
| Learning rate | `1e-6` | `2e-5` | Cấu hình local dùng LR cao hơn paper. |
| Optimizer | AdamW | AdamW | Giữ nguyên. |
| LR schedule | Exponential decay `0.9` mỗi 100 iteration | Có exponential decay trong code; log thấy scheduler step tại optimizer step 100 | Cùng kiểu schedule, khác LR ban đầu. |
| Gradient accumulation | Không nhấn mạnh trong paper | `gradient_accumulation_steps=4` | Dùng để ổn định update trong điều kiện local. |
| Retriever temperature | `0.1` | `temperature_r=0.1` | Giữ nguyên. |
| LM temperature | Không phải điểm chính trong paper | `temperature_lm=1.0` | Theo config local. |
| Precision retriever | Không nêu như ràng buộc phần cứng | `retrieval_use_fp16=True` | Tiết kiệm VRAM trên RTX 2080 Ti. |
| Early stopping | Dừng khi validation loss không cải thiện trong một epoch | Bật, patience `5`, min delta `1e-4` | Giữ ý tưởng, nhưng validation local bị giới hạn. |
| Validation frequency | Không nêu cùng dạng flag | Mỗi `100` train step | Theo code/config local. |
| Validation batch size | Không nêu | `2` | Giảm để tránh OOM. |
| Validation max batches | Ngầm hiểu full validation | `10` batch | Giảm thời gian và VRAM. |

## Hyperparameter Retrieval

| Hyperparameter | Paper ReSCORE | Reproduction cục bộ | Thay đổi / Lý do |
|---|---:|---:|---|
| Số document candidate khi train `M` | `32` | `retrieval_buffer_size=4` | Giảm mạnh để tránh OOM và giảm chi phí scoring. |
| Top-k khi inference | `8` | train hiện tại `retrieval_count=4`; inference cũng đề xuất `4` | Giảm để vừa VRAM/context. |
| Retrieval batch size | Không phải thông số chính trong paper | `64` | Tối ưu để dùng GPU retriever mà không OOM. |
| Retrieval query type | Query iterative từ question/thought | `full` | Dùng toàn bộ trạng thái query trong code local. |
| Retrieval training strategy | Train query embedder | `query_only` | Khớp với ý tưởng train query encoder. |
| Loại bỏ duplicate retrieval | Không nhấn mạnh | Mặc định theo code nếu không bật flag | Cần ghi rõ nếu bật trong các run sau. |

## Hyperparameter Generation Và vLLM

| Hyperparameter | Paper ReSCORE | Reproduction cục bộ | Thay đổi / Lý do |
|---|---:|---:|---|
| Backend generation | Llama trong setup paper | External `vllm_server` khi train | Tránh load Llama trực tiếp trong process train retriever. |
| Load vLLM trong train | Không áp dụng | `local_vllm_load=False` | Train gọi API server. |
| Generation max batch size | Không nêu như ràng buộc local | `1` | Giảm để tránh OOM. |
| Generation max total tokens | Paper không giảm theo cách này | `1536` | Giảm độ dài context cho GPU 10.75GiB. |
| Generation max new tokens | Không phải thông số chính | `48` | Giảm độ dài sinh để ổn định và nhanh hơn. |
| Scoring mode | Xác suất LLM cho question/answer | `prompt_logprobs` | Dùng vì echo logprobs của vLLM gây lỗi server. |
| vLLM score max tokens | Không nêu | `96` | Cắt ngắn target scoring để ổn định. |
| vLLM prompt logprobs | Không nêu | `1` | Lấy logprob tối thiểu phục vụ scoring. |
| Missing logprob fallback | Không nêu | `-20.0` | Tránh crash khi vLLM thiếu token logprob. |

## Hyperparameter Iterative Reasoning

| Hyperparameter | Paper ReSCORE | Reproduction cục bộ | Thay đổi / Lý do |
|---|---:|---:|---|
| Số iteration tối đa | `6` | `max_num_thought=2` | Giảm runtime, prompt length và OOM risk. |
| Số iteration tối thiểu | `2` | Chưa đảm bảo giống paper nếu code/config không ép min iteration | Cần ghi là khác biệt implementation. |
| Query reformulation | Thought-based iterative retrieval trong IQATR | Pipeline local dùng iterative state và `retrieval_query_type=full` | Giữ ý tưởng iterative, nhưng chưa reproduce ablation chính xác. |
| Prompt max paragraph count | Paper không nêu dạng giới hạn giảm | `8` | Giảm prompt length. |
| Prompt max paragraph words | Paper không nêu dạng giới hạn giảm | `220` | Giữ prompt dưới context limit của server/model. |

## Checkpoint Và Resume

| Hạng mục | Paper | Reproduction cục bộ | Ghi chú |
|---|---:|---:|---|
| Tính liên tục của training | Một run sạch theo setup paper | Nhiều lần resume | Do OOM và reboot server. |
| Checkpoint được chọn | Best model từ training paper | `epoch_0_step_200` từ run MuSiQue đã resume | Chọn theo validation loss thấp nhất local. |
| Lưu checkpoint | Không nêu | Docker-backed `predictions` symlink | Tránh nghẽn I/O trên NFS. |
| Checkpoint thừa | Không áp dụng | Đã xoá, chỉ giữ checkpoint được chọn | Dễ inference và báo cáo. |

## Câu Mô Tả Đề Xuất Cho Báo Cáo

Có thể dùng đoạn sau trong báo cáo:

```text
Chúng tôi reproduce ReSCORE trong điều kiện phần cứng hạn chế thay vì cố gắng khớp hoàn toàn với paper. Paper sử dụng hai GPU NVIDIA A100 40GB, trong khi môi trường cục bộ sử dụng GPU RTX 2080 Ti-class với khoảng 10.75GiB VRAM. Để training khả thi, chúng tôi giữ nguyên ý tưởng cốt lõi của ReSCORE: sử dụng supervision relevance-consistency từ LLM để train query encoder của Contriever. Tuy nhiên, chúng tôi giảm số document candidate khi train, số document retrieve khi inference, độ dài prompt, độ dài context generation, generation batch size và số reasoning iteration tối đa. Ngoài ra, chúng tôi dùng external vLLM server cho Llama-3.1-8B-Instruct và lưu checkpoint/log xuống Docker local storage để tránh nghẽn I/O trên NFS.
```

## Metric Tối Thiểu Cần Báo Cáo Với Setup Local

| Metric | Có cần báo cáo? | Ghi chú |
|---|---:|---|
| Answer EM | Có | Exact match ở mức answer, theo official/internal evaluator. |
| Answer F1 | Có | Metric chính cho chất lượng answer. |
| MHR_1@8 | Nên có | Cần retrieval trace/support labels. |
| MHR_2@8 | Nên có | Quan trọng vì local dùng `max_num_thought=2`. |
| MHR_final@8 | Nên có | Multi-hop recall tích luỹ ở iteration cuối. |
| Runtime / hardware note | Có | Cần nêu rõ vì setup khác paper đáng kể. |
