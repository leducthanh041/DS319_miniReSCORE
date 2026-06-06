# ReSCORE Pipeline for Reproduction

> Phạm vi: tài liệu này chỉ dựa trên 3 nguồn bạn cung cấp: `2025.acl-long.16.pdf`, `SomeinformationReSCORE.md`, và source code trong `ReSCORE-main.zip`.
> 
> Nguyên tắc đọc: tôi ưu tiên **paper làm chuẩn khái niệm** và dùng **source code để xác định pipeline thực thi thật sự**. Ở đâu paper và code không khớp, tôi ghi rõ là **khác biệt / chưa đủ bằng chứng**.

## 1. Tổng quan mục tiêu của ReSCORE

### ReSCORE giải quyết bài toán gì
- ReSCORE nhắm vào bài toán **Multi-hop Question Answering (MHQA)**: trả lời một câu hỏi cần ghép thông tin từ nhiều tài liệu qua nhiều bước suy luận.
- Trở ngại chính mà paper nêu ra là:
  - dense retriever thường tốt hơn BM25,
  - nhưng dense retriever cần cặp **query-document labels** để fine-tune,
  - còn trong MHQA thì query thay đổi theo từng vòng lặp suy luận nên việc gán nhãn là rất đắt.

### Ý tưởng cốt lõi của phương pháp
- ReSCORE huấn luyện retriever **không cần document labels thủ công**.
- Thay vào đó, paper dùng **LLM làm teacher** để sinh **pseudo-GT distribution** cho tài liệu.
- Điểm pseudo-GT dựa trên 2 yếu tố:
  - **Relevance**: tài liệu có liên quan tới câu hỏi không.
  - **Consistency**: tài liệu có nhất quán với câu trả lời đúng không.
- Retriever được train để phân phối truy xuất của nó gần với phân phối pseudo-GT đó.
- Hệ thống QA dùng retriever đã được train gọi là **IQATR**.

### Cách hiểu ngắn gọn
- **Paper-level view**: ReSCORE = cách train retriever.
- **System-level view**: IQATR = iterative RAG framework dùng retriever đã train bởi ReSCORE.

---

## 2. Pipeline tổng thể

### 2.1. Pipeline khái niệm theo paper

Paper mô tả một iterative RAG loop:

```text
Question q
  -> Retrieve top-k docs D(1)
  -> LLM answer generation
      -> nếu answer != "unknown": stop
      -> nếu answer == "unknown":
           generate thought t(1)
           reformulate query q(2)
           retrieve D(2)
           ... lặp tiếp ...
```

### 2.2. Hai pipeline cần tách riêng để reproduce

#### A. Inference pipeline
```text
question
  -> build retrieval query
  -> dense retrieval from FAISS index
  -> answer generation
  -> if not enough: thought generation
  -> update retrieval query
  -> repeat until stop
  -> final answer
```

#### B. Retriever training pipeline
```text
question + GT answer
  -> iterative retrieval of top-M docs
  -> for each retrieved doc: compute LLM-based pseudo-GT score
  -> compute retriever score by query-doc dot product
  -> KL(Q_LM || P_R)
  -> backprop to query encoder
  -> continue along iterative paths
```

### 2.3. Pipeline thực thi thật sự trong repo

#### Training pipeline được lắp trong code
`source/run/train.py:128-166`

```text
QuestionState
  -> RetrievalStep
  -> TrainStep
  -> GenerationStep(answer)
  -> EndStep
  -> GenerationStep(thought)
  -> quay lại RetrievalStep cho vòng sau
```

#### Inference pipeline được lắp trong code
`source/run/inference.py:52-69`

```text
QuestionState
  -> RetrievalStep
  -> GenerationStep(answer)
  -> EndStep
  -> GenerationStep(thought)
  -> quay lại RetrievalStep cho vòng sau
```

### 2.4. Điểm quan trọng
- Trong **paper**, query reformulation là một bước khái niệm rõ ràng.
- Trong **repo hiện có**, không có step riêng để sinh `q^(i+1)` bằng prompt rewrite trong active pipeline.
- Query update trong code được thực hiện gián tiếp bằng cách **nối question với accumulated thoughts** trong `preprocess_retrieval_query()`.

---

## 3. Inference pipeline

## 3.1. Input

### Dữ liệu đầu vào trong code
- Inference đọc từ `cfg.data_file_path` trong `PipelineConfig`.
- Với `dataset_split != train`, file được kỳ vọng là:
  - `./data/processed_data/{dataset}/dev_subsampled.jsonl`
  - hoặc `./data/processed_data/{dataset}/test_subsampled.jsonl`
- Hàm đọc dữ liệu là `load_data_from_jsonl()` trong `source/utility/data_utils.py:27-70`.

### Format mỗi sample
Repo kỳ vọng mỗi dòng JSONL có ít nhất các field:
- `question_id`
- `question_text`
- `answers_objects`
- `contexts`

### State khởi tạo
- Mỗi câu hỏi được bọc thành `QuestionState(question_id, question)`.
- Định nghĩa state nằm ở `source/pipeline/state.py:26-36`.

## 3.2. Retrieval

### Thành phần chính
- `RetrievalStep` ở `source/pipeline/step/retrieval.py:59-119`
- Retriever encoder ở `source/module/retrieve/dense.py:43-165`
- FAISS index ở `source/module/index/index.py`

### Query được tạo như thế nào
Code không sinh một câu hỏi rewrite riêng. Thay vào đó, query retrieval được tạo bởi:
- `source/pipeline/utils.py:90-114`

Có 2 mode:
- `retrieval_query_type='last_only'`
  - nếu đã có thought thì lấy thought cuối,
  - nếu chưa có thought thì lấy question gốc.
- `retrieval_query_type='full'`
  - ghép `question + all thoughts` bằng ký tự newline.

### Điều này có nghĩa gì
- **Repo active path = thought-based query accumulation**.
- Nó gần với **Thought-concat** trong paper hơn là **LLM-rewrite**.
- Nhưng code ghép theo dạng:
  - `question` trước,
  - rồi `all thoughts` sau.
- Trong paper, Thought-concat được mô tả là `q(i+1) = [t(i); q(i)]`.
- Vì vậy, **không thể kết luận code đang hiện thực đúng từng ký hiệu của paper**; chỉ có thể nói code đang dùng một biến thể thought concatenation.

### Truy xuất tài liệu
- Query được embed bằng query encoder.
- FAISS search lấy ra `retrieval_buffer_size` document candidates.
- Sau đó `filter_document()` bỏ duplicate nếu bật `retrieval_no_duplicates`.
- Kết quả được đưa vào `DocumentState`.

### K và M trong code
Có 2 số lượng tài liệu khác nhau:
- `retrieval_buffer_size`
  - số tài liệu lấy từ index,
  - đóng vai trò **top-M** dùng trong training loss.
- `retrieval_count`
  - số tài liệu thật sự đưa vào prompt answer/thought,
  - đóng vai trò **top-k** dùng cho inference reasoning.

## 3.3. Answer generation

### Code path
- Prompt generator: `AnswerGeneratePromptGenerator` trong `source/pipeline/step/generation.py:185-220`
- Output parser: `AnswerGenerateOutputParser` trong `source/pipeline/step/generation.py:93-127`
- Prompt file đang dùng: `prompts/prompt_set__1/answer_gen.txt`

### Prompt nhận gì
Prompt answer trong repo nhận:
- `documents`
- `question`
- `thoughts`

### Output format
LLM phải trả JSON:
```json
{"answer": "..."}
```

### Cách parser xử lý
- Nếu parse được JSON và có key `answer` -> tạo `AnswerState`.
- Nếu không parse được -> ép thành `answer='unknown'`.
- Ngoài ra parser còn có gate `min_num_thought`:
  - nếu số thought hiện có nhỏ hơn `cfg.min_num_thought`, parser trả luôn `unknown`.

### Ý nghĩa của `min_num_thought`
- `PipelineConfig.__post_init__()` đặt:
  - `method='iqatr'` -> `min_num_thought = 1`
  - `method='base'` -> `min_num_thought = 0`
- Nếu `min_num_thought = 1`, hệ thống buộc phải đi qua ít nhất 1 thought trước khi cho phép answer khác `unknown`.
- Đây là cách repo cố gắng ép **ít nhất 2 hops**.

## 3.4. Thought generation

### Code path
- Prompt generator: `ThoughtGeneratePromptGenerator` trong `source/pipeline/step/generation.py:222-257`
- Output parser: `ThoughtGenerateOutputParser` trong `source/pipeline/step/generation.py:131-156`
- Prompt file: `prompts/prompt_set__1/thought_gen.txt`

### Vai trò
- Tóm tắt thông tin vừa retrieve được thành một câu ngắn.
- Thought sau đó được lưu trong `ThoughtState`.
- Ở vòng tiếp theo, thought đi vào retrieval query và cũng đi vào answer/thought prompt.

## 3.5. Query reformulation

### Theo paper
Paper có 2 kiểu reformulation trong phần phân tích:
- **LLM-rewrite**: LLM viết lại câu hỏi mới.
- **Thought-concat**: nối thought vào query để retrieve tiếp.

### Trong repo active pipeline
- **Không có một `QuestionRewriteStep` riêng**.
- **Không có prompt rewrite nào trong thư mục prompts đang được active sử dụng**.
- Query mới được hình thành hoàn toàn thông qua `preprocess_retrieval_query()`.

### Kết luận phần này
- Nếu bạn reproduce **theo paper đầy đủ**, cần phân biệt 2 biến thể reformulation.
- Nếu bạn reproduce **theo repo release hiện có**, active path đang bám theo **thought accumulation**, không phải LLM rewrite prompt như trong appendix E.3.

## 3.6. Điều kiện dừng lặp

### EndStep
`source/pipeline/step/end.py:58-125`

Hệ thống dừng khi:
- có bất kỳ `AnswerState.answer.lower() != 'unknown'`, hoặc
- số thoughts đã đạt `max_num_thought`.

### Nếu đạt `max_num_thought`
- Repo kết thúc bằng cách ghép toàn bộ thoughts lại thành `answer` fallback.
- Đây là hành vi kỹ thuật của code.
- Paper mô tả dừng khi answer không còn là `unknown`; phần fallback bằng cách ghép thoughts là chi tiết cài đặt của repo.

### Demo app
- `demo/app.py` còn có thêm logic thủ công: nếu 1-hop đã ra answer thì vẫn ép trả về `Unknown` để đi tiếp một hop xác minh.
- Đây là logic demo UI, không phải mô tả phương pháp trong paper.

---

## 4. Retriever training pipeline

## 4.1. Cách tạo pseudo-GT labels

### Theo paper
Paper định nghĩa:
- `Q_LM^(i)(d_j^(i) | q^(i)) ∝ P_LM^(i)(a, q | d_j^(i))`
- và factorize thành:
  - `P_LM(q | d)`  -> relevance
  - `P_LM(a | q, d)` -> consistency

Ý tưởng:
- document tốt phải vừa liên quan tới question,
- vừa hỗ trợ answer đúng.

### Theo code đang chạy
Pseudo-GT trong repo được hiện thực ở `TrainStep`:
- file: `source/pipeline/step/training.py:68-167`

Quy trình:
1. RetrievalStep lấy `N = retrieval_buffer_size` docs.
2. Với mỗi doc, code tạo một prompt từ `qa_gen_input.txt`.
3. Target output là JSON question-answer cố định từ `qa_gen_output.txt`.
4. `generator.score(prompts, answers)` tính perplexity score.
5. `softmax(-lm_score / temperature_lm)` tạo phân phối pseudo-GT.

### Điều quan trọng cần đọc kỹ
Trong repo:
- `qa_gen_input.txt` là prompt **question-answer pair generation from document(s)**.
- Prompt file này **chỉ condition trên document(s)**.
- `qa_gen_output.txt` chèn **question gốc** và **GT answer** làm target string.

Nói cách khác, repo đang hiện thực gần nhất với:
- **PLM(q, a | d)**

và **không thấy active implementation riêng** cho:
- `PLM(q | d)`
- `PLM(a | q, d)`

mặc dù paper có mô tả 3 prompt này trong phần appendix/phân tích.

## 4.2. Ý nghĩa của relevance và consistency

### Relevance
- Tài liệu có đủ chủ đề / thực thể / quan hệ để khớp với câu hỏi hay không.
- Paper nhấn mạnh rằng nếu chỉ nhìn answer consistency thì dễ dính false positive kiểu tài liệu chứa đúng token answer nhưng sai ngữ cảnh.

### Consistency
- Với question đã cho, tài liệu có thực sự hỗ trợ answer đúng hay không.

### Kết hợp hai yếu tố
- Paper chọn `PLM(q,a|d)` làm default pseudo-GT vì nó đồng thời kiểm tra:
  - topical relevance,
  - answer support.

## 4.3. Phân phối `Q_LM` và `P_R`

### `Q_LM`
- Paper: pseudo-GT distribution do LLM tạo trên các document candidates.
- Code: được xấp xỉ bằng
  - `lm_score = generator.score(prompts, answers)`
  - `lm_likelihood = softmax(-lm_score / temperature_lm)`

### `P_R`
- Paper: retriever distribution từ dot product giữa query embedding và document embedding.
- Code:
  - query embedding từ `retriever.embed(..., input_type='query')`
  - document embedding đọc trực tiếp từ FAISS/docstore index qua `indexer.get_embedding_from_docstore_id(...)`
  - `r_score = sum(query_emb * doc_emb)`
  - `retriever_likelihood = log_softmax(r_score / temperature_r)`

## 4.4. KL divergence training objective

### Paper
- Mục tiêu: minimize `DKL(Q_LM || P_R)` qua từng iteration và từng QA pair.

### Code
Trong `source/pipeline/step/training.py:154-167`:
- `retriever_likelihood = log_softmax(...)`
- `lm_likelihood = softmax(...)`
- `loss = F.kl_div(retriever_likelihood, lm_likelihood, reduction='batchmean')`
- sau đó chia tiếp cho `gradient_accumulation_steps`

=> Về mặt tinh thần, code khớp với objective KL divergence mà paper mô tả.

## 4.5. Quy trình training theo iteration

### Theo paper
- Training diễn ra trong iterative RAG process.
- Số iteration phụ thuộc việc answer còn `unknown` hay không.

### Theo code
`PipelineController.train()` lặp trên state paths:
1. `RetrievalStep`
2. `TrainStep` tính loss trên batch current paths
3. `GenerationStep(answer)`
4. `EndStep`
5. `GenerationStep(thought)`
6. quay lại vòng tiếp nếu chưa end

### Cách nhìn chính xác hơn
- Loss không chỉ tính một lần cho toàn câu hỏi.
- Nó được tính **ở mỗi iteration đang sống** trong state path.
- `PipelineController.train()` cộng loss theo step rồi lấy trung bình.

## 4.6. Khác biệt paper vs code ở training

### Khác biệt 1: pseudo-GT active path trong repo chỉ thấy `PLM(q,a|d)`
- Paper phân tích 3 kiểu pseudo-GT.
- Repo active prompts chỉ đủ bằng chứng cho kiểu `PLM(q,a|d)`.

### Khác biệt 2: pseudo-GT prompt trong repo không thực sự dùng `thoughts`
- `TrainStep.formatting()` có truyền `question`, `thoughts`, `documents` vào `.format(...)`.
- Nhưng `qa_gen_input.txt` hiện tại chỉ dùng `{documents}`.
- Vì vậy prompt pseudo-GT của repo **không trực tiếp condition trên current thoughts/query string**.
- Tác động của iteration trong code chủ yếu đến từ:
  - tập documents được retrieve ở iteration đó,
  - query embedding ở phía retriever.

### Khác biệt 3: paper notation là theo `q^(i)`
- Paper viết pseudo-GT theo query của từng iteration.
- Repo không cho thấy một prompt teacher thực sự chứa `q^(i)` như một trường nhập riêng trong active pseudo-GT file.
- Tôi không có đủ bằng chứng để khẳng định repo đã hiện thực đúng hoàn toàn theo notation đó.

---

## 5. Mapping giữa bài báo và source code

| Paper component | Code location | Chức năng / ghi chú |
|---|---|---|
| Iterative RAG controller | `source/pipeline/controller.py` | Quản lý state tree, chạy pipeline theo từng path, lưu kết quả cuối. |
| Question / Thought / Document / Answer / End state | `source/pipeline/state.py` | Biểu diễn toàn bộ loop suy luận dưới dạng state machine. |
| Retrieval step | `source/pipeline/step/retrieval.py` | Tạo retrieval query, embed query, search FAISS index, filter duplicate docs. |
| Retrieval query update | `source/pipeline/utils.py::preprocess_retrieval_query` | Hiện thực active query reformulation trong repo bằng question + thoughts. |
| Dense retriever | `source/module/retrieve/dense.py` | Query encoder / passage encoder, mean pooling, embed query/passage. |
| FAISS index + docstore | `source/module/index/index.py`, `source/module/index/docstore.py` | Lưu embedding, tìm kiếm inner-product, map FAISS id -> document. |
| Answer generation | `source/pipeline/step/generation.py` + `prompts/prompt_set__1/answer_gen.txt` | Tạo prompt answer và parse JSON answer. |
| Thought generation | `source/pipeline/step/generation.py` + `prompts/prompt_set__1/thought_gen.txt` | Tạo prompt thought và parse JSON thought. |
| End condition | `source/pipeline/step/end.py` | Dừng khi answer khác `unknown` hoặc đạt `max_num_thought`. |
| ReSCORE loss | `source/pipeline/step/training.py` | Tính retriever score, teacher LM score, rồi KL divergence. |
| Pseudo-GT prompt active | `prompts/prompt_set__1/qa_gen_input.txt` + `qa_gen_output.txt` | Hiện thực gần nhất với `PLM(q,a|d)` trong repo. |
| Inference entrypoint | `source/run/inference.py` | Lắp pipeline suy luận, chạy evaluation sau inference. |
| Training entrypoint | `source/run/train.py` | Lắp training pipeline, optimizer, validation, save checkpoint. |
| Build passage corpus TSV | `source/run/preprocess_raw_data.py` | Chuyển raw wiki/context thành TSV để embed. |
| Generate passage embeddings | `source/run/generate_passage_embeddings.py` | Encode passages thành embedding và dump shard pickle. |
| Build FAISS index | `source/run/build_index.py` | Nạp các shard embedding rồi build/save index. |
| Dataset JSONL loader | `source/utility/data_utils.py` | Đọc processed_data JSONL, tạo ground-truth answer dict. |
| Raw download scripts | `script/download/*.sh` | Tải raw data, sampled processed data, build database. |

### Mapping theo luồng thực tế

#### Inference
1. `source/run/inference.py`
2. `PipelineController`
3. `RetrievalStep`
4. `GenerationStep(answer)`
5. `EndStep`
6. nếu chưa dừng -> `GenerationStep(thought)`
7. `preprocess_retrieval_query()` tạo query cho vòng sau

#### Training
1. `source/run/train.py`
2. `PipelineController.train()`
3. `RetrievalStep`
4. `TrainStep`
5. `GenerationStep(answer)`
6. `EndStep`
7. nếu chưa dừng -> `GenerationStep(thought)`
8. lặp tiếp cho đến khi path kết thúc

---

## 6. Các prompt chính trong hệ thống

## 6.1. Answer generation prompt

### Paper
- Appendix E.1 mô tả prompt answer generation.
- Input: documents, question, hints/thoughts.
- Output: JSON `{"answer": ...}` hoặc `"Unknown"`.

### Repo
- File active: `prompts/prompt_set__1/answer_gen.txt`
- Input fields trong prompt:
  - `Documents:`
  - `Question:`
  - `Thoughts:`
- Parser kỳ vọng key `answer`.

## 6.2. Thought generation prompt

### Paper
- Appendix E.2 mô tả prompt thought generation.
- Vai trò: sinh một câu tóm tắt partial information để hỗ trợ hop sau.

### Repo
- File active: `prompts/prompt_set__1/thought_gen.txt`
- Output: JSON `{"thought": ...}`

## 6.3. Question rewriting prompt

### Paper
- Appendix E.3 có prompt rewrite riêng cho **LLM-rewrite**.

### Repo
- Tôi **không thấy prompt rewrite active tương ứng** trong thư mục `prompts/prompt_set__1/`.
- Tôi cũng **không thấy một step rewrite riêng** trong pipeline đang được lắp ở `train.py` và `inference.py`.

### Kết luận
- Question rewriting là **thành phần có trong paper/appendix**.
- Nhưng trong repo release hiện có, active pipeline chủ yếu dùng **thought accumulation** thay cho rewrite prompt riêng.

## 6.4. Prompt dùng cho pseudo-GT

### Paper
Paper nhắc tới 3 loại prompt để đánh giá document:
- `PLM(a | q, d)`
- `PLM(q | d)`
- `PLM(q, a | d)`

### Repo
Active files tôi thấy:
- `prompts/prompt_set__1/qa_gen_input.txt`
- `prompts/prompt_set__1/qa_gen_output.txt`

### Chức năng thật sự
- `qa_gen_input.txt`: condition prompt từ document(s)
- `qa_gen_output.txt`: target JSON chứa question và answer

=> Đây là hiện thực gần nhất của **`PLM(q, a | d)`**.

### Điều tôi không thấy trong repo
- Không thấy prompt file active cho:
  - `PLM(a | q, d)` riêng
  - `PLM(q | d)` riêng
- Không đủ bằng chứng để nói repo release hiện tại có sẵn full ablation code cho cả 3 loại prompt.

## 6.5. Prompt files khác trong repo
Có thêm:
- `multi_retr_answer_direct_gen.txt`
- `multi_retr_thought_direct_gen.txt`

Nhưng trong pipeline active ở `train.py` / `inference.py`, các file đang được dùng trực tiếp vẫn là:
- `answer_gen.txt`
- `thought_gen.txt`
- `qa_gen_input.txt`
- `qa_gen_output.txt`

---

## 7. Những điểm cần chú ý nếu muốn reproduce

## 7.1. Dữ liệu

### Những gì inference/train thực sự cần
Repo cần đồng thời 2 loại dữ liệu:

#### A. Processed QA data
Để train/infer, code cần:
- `./data/processed_data/{dataset}/train.jsonl`
- `./data/processed_data/{dataset}/dev_subsampled.jsonl`
- `./data/processed_data/{dataset}/test_subsampled.jsonl`

#### B. Retrieval database
Để retrieve, code cần index tại:
- `./data/database/contriever_msmarco/{dataset}`

### Vấn đề reproduction quan trọng
README phần `Data Preparation` chỉ nói:
- download raw data
- build retrieval DB

Nhưng như vậy **chưa đủ** cho train/inference, vì:
- `processed_data/*.jsonl` không được `build.sh` tạo ra,
- inference lại đọc `*_subsampled.jsonl`.

### Để reproduce an toàn
Bạn cần ít nhất một trong hai cách:

#### Cách 1: dùng processed_data đã sample sẵn
- chạy `script/download/multihop_sampled_data.sh`
- cách này cho bạn `processed_data/.../*.jsonl` phục vụ dev/test subsampled

#### Cách 2: tự preprocess từ raw
- chạy các script trong `preprocess/`:
  - `process_hotpotqa.py`
  - `process_musique.py`
  - `process_2wikimultihopqa.py`
- rồi chạy tiếp `preprocess/subsample_dataset_and_remap_paras.py` nếu cần subsampled splits

## 7.2. Retriever

### Backbone
- Repo dùng `facebook/contriever-msmarco` làm dense retriever base.
- Document encoder và query encoder ban đầu cùng checkpoint.

### Khi train
- Paper nói chỉ train **question embedder** và freeze document embedder.
- Repo khớp ý này khi dùng `training_strategy='query_only'`.
- Trong `DenseRetriever`, passage model được copy từ query model rồi để `eval()`.

### Retrieval store
- Passage embeddings được precompute.
- Sau đó build FAISS inner-product index.
- Trong training, doc embeddings không được recompute online từ passage encoder; code lấy trực tiếp từ index/docstore.

## 7.3. LLM

### Backbone generation/scoring
- Paper dùng `Llama-3.1-8B-Instruct`.
- Repo cũng dùng Llama 3.1 cho generation/scoring.

### Train vs inference
- `train.py` khởi tạo `LlamaGenerator(..., use_vllm=False)`
- `inference.py` khởi tạo `LlamaGenerator(..., use_vllm=True)`

### Ý nghĩa
- Training cần scoring teacher probabilities/perplexity bằng HF model path trong code.
- Inference ưu tiên throughput bằng vLLM.

## 7.4. Hyperparameters

### Hyperparameters paper báo cáo
- top-M cho training distribution: **32**
- top-k cho inference answering: **8**
- max iterations: **6**
- minimum iteration limit: **2**
- batch size: **16**
- temperature scaling: **0.1**
- optimizer: **AdamW**
- lr: **1e-6**
- lr decay: exponential, factor **0.9** mỗi 100 steps

### Hyperparameters / defaults trong repo
Có vài chỗ không khớp paper hoàn toàn:

| Mục | Paper | Repo default / active evidence | Nhận xét |
|---|---:|---:|---|
| training top-M | 32 | `train.py --retrieval_buffer_size=32` | Khớp nếu dùng CLI train mặc định. |
| inference top-k | 8 | `retrieval_count=8` | Khớp. |
| max iter | 6 | `max_num_thought=6` | Khớp theo code logic. |
| min 2-hop | có | phụ thuộc `method='iqatr'` | Không tự động đúng nếu dùng default sai. |
| batch size | 16 | `train.py` default 20 | Khác paper. |
| lr | 1e-6 | `train.py` default 2e-5 | Khác paper. |
| prompt set | appendix prompt set đang dùng | `train.py` default 26 | Repo chỉ có `prompt_set__1`, nên default 26 là lệch. |

## 7.5. Iterative setting

### Minimum 2-hop trong paper
Paper nói họ đặt minimum iteration limit = 2 ở training và inference.

### Trong repo
- Điều này được encode gián tiếp bởi `min_num_thought` trong `PipelineConfig`.
- Nhưng điều này chỉ đúng nếu:
  - `method='iqatr'`
- Nếu `method='base'` thì `min_num_thought=0`.

### Vấn đề thực tế
- `train.py` parse default lại đặt `--method rescore`.
- Trong `PipelineConfig.__post_init__()`, giá trị `rescore` không được handle như một case chuẩn.
- Kết quả là nó rơi vào nhánh `Not Implemented` và `min_num_thought=0`.

### Hệ quả
- Nếu bạn chạy train đúng như README mà không sửa thêm, behavior có thể **không khớp** với mô tả minimum 2-hop của paper.
- Muốn bám paper hơn, bạn nên kiểm tra / sửa `method` để dùng logic tương đương `iqatr`.

## 7.6. Những dependency logic quan trọng trong code

### 1. `prompt_set`
- `train.py` default `prompt_set=26`
- nhưng repo chỉ có `prompts/prompt_set__1/`
- Nếu không override, bạn rất dễ lỗi missing prompt files.

### 2. `inference.py` import sai
- File đang import:
  - `from source.pipeline.step.__retrieval import RetrievalStep`
- Trong repo không có file `__retrieval.py`.
- File đúng đang tồn tại là `source/pipeline/step/retrieval.py`.
- Đây là lỗi thực thi cần sửa trước khi chạy inference.

### 3. Data preparation trong README chưa đủ kín
- Cần cả `processed_data` lẫn `database`.
- README chỉ nói raw data + build DB, nên nếu chạy y nguyên rất dễ thiếu QA jsonl.

### 4. `generate_passage_embeddings.py` có chi tiết cần lưu ý
- Dù parser nhận `--model_name_or_path`, config trong code lại hardcode `facebook/contriever-msmarco`.
- Nghĩa là script hiện tại không thật sự linh hoạt theo arg như tên gọi gợi ý.

### 5. Query reformulation trong repo là logic-level, không phải step-level
- Nếu bạn muốn reproduce đúng bản paper có LLM rewrite ablation,
- bạn có thể phải tự bổ sung step/prompt rewrite riêng vì active code chưa cho thấy thành phần đó.

### 6. Teacher scoring thực chất dùng perplexity của target string
- `LlamaGenerator.score()` tính perplexity của output target given prompt.
- Vì vậy pseudo-GT trong code là một **scoring-by-likelihood of forced output**, không phải một module ranker riêng.

---

## 8. Tóm tắt ngắn gọn để bắt đầu reproduce

## 8.1. Các bước cần làm theo thứ tự

### Bước 1: đọc paper phần lõi
Đọc theo thứ tự:
1. Sec. 3.1: Iterative RAG Framework
2. Sec. 3.2: Training Retriever for Iterative RAG
3. Sec. 4.1: Implementation Details
4. Appendix B + E: prompts

### Bước 2: đọc repo theo đúng luồng chạy
Đọc theo thứ tự:
1. `README.md`
2. `source/run/train.py`
3. `source/run/inference.py`
4. `source/pipeline/controller.py`
5. `source/pipeline/step/retrieval.py`
6. `source/pipeline/step/training.py`
7. `source/pipeline/step/generation.py`
8. `source/pipeline/step/end.py`
9. `source/pipeline/utils.py`
10. `prompts/prompt_set__1/*`

### Bước 3: chuẩn bị data
- tải raw data nếu cần
- tạo hoặc tải `processed_data/*.jsonl`
- build retrieval DB (`embed_ready_data` -> embeddings -> FAISS index)

### Bước 4: sửa các điểm dễ lỗi trước khi chạy
Tối thiểu nên kiểm tra:
- `inference.py` import retrieval path
- `prompt_set`
- `method`
- đường dẫn database / processed_data đã tồn tại chưa

### Bước 5: build retrieval database
Luồng tương ứng:
1. `source/run/preprocess_raw_data.py`
2. `source/run/generate_passage_embeddings.py`
3. `source/run/build_index.py`

### Bước 6: train retriever
- chạy `source/run/train.py`
- nhưng nên chỉnh config để bám paper hơn:
  - `prompt_set=1`
  - `method=iqatr` hoặc logic tương đương
  - kiểm tra lại batch size / lr nếu muốn khớp paper

### Bước 7: run inference
- sửa import lỗi trong `inference.py`
- dùng checkpoint query encoder đã train
- chạy iterative inference loop

## 8.2. Tôi nên đọc file nào trước, chạy phần nào trước

### Nên đọc trước
1. `2025.acl-long.16.pdf`
2. `README.md`
3. `source/run/train.py`
4. `source/pipeline/step/training.py`
5. `source/pipeline/step/retrieval.py`
6. `source/pipeline/utils.py`
7. `prompts/prompt_set__1/qa_gen_input.txt`
8. `prompts/prompt_set__1/answer_gen.txt`
9. `prompts/prompt_set__1/thought_gen.txt`

### Nên chạy theo thứ tự
1. data preparation
2. build index
3. kiểm tra inference pipeline với base retriever
4. train retriever bằng ReSCORE
5. inference lại với retriever đã train
6. evaluation

## 8.3. Kết luận cuối cùng để reproduce đúng

Nếu mục tiêu của bạn là **hiểu đủ rõ để reproduce**, thì cách an toàn nhất là tách 2 lớp:

### Lớp 1: reproduce theo paper
- hiểu đầy đủ khái niệm `Q_LM`, `P_R`, KL loss, iterative RAG, query reformulation, pseudo-GT ablations.

### Lớp 2: reproduce theo repo release
- active code hiện tại tương ứng mạnh nhất với:
  - iterative retrieval + answer/thought loop
  - thought-based query accumulation
  - pseudo-GT kiểu `PLM(q,a|d)`
  - query encoder training via KL divergence

### Một câu chốt
- **Paper mô tả framework tổng quát hơn.**
- **Repo release hiện tại là một hiện thực cụ thể, thiên về Thought-concat + `PLM(q,a|d)` active path, và có vài chi tiết cần sửa/override trước khi chạy thật.**
