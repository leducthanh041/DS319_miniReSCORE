# ReSCORE AGENTS.md

## Scope

Thư mục này là repo riêng cho project `DS319/ReSCORE`. Mọi tìm kiếm, chỉnh sửa và
chạy lệnh mặc định phải ở trong repo này, không lan sang các project khác trong
workspace `Thanhld`.

## Source Of Truth

Ưu tiên đọc theo thứ tự:

1. File này
2. `ReSCORE_Pipeline_for_Reproduction.md`
3. `README.md`
4. Source code đang chạy thật trong `source/`

Khi paper, ghi chú reproduce và code không hoàn toàn khớp nhau:

- dùng `source/` làm chuẩn cho execution path thực tế,
- dùng `ReSCORE_Pipeline_for_Reproduction.md` để giải thích pipeline và chỉ ra
  khác biệt giữa paper và repo release.

## Project Focus

Agent làm việc trong repo này nên mặc định hiểu đây là project về:

- multi-hop question answering
- iterative retrieval + answer/thought generation
- retriever training theo ReSCORE
- reproduction, debugging, evaluation và prompt/pipeline analysis

## Key Entry Points

- Training: `source/run/train.py`
- Inference: `source/run/inference.py`
- Pipeline config: `source/pipeline/config.py`
- Pipeline steps: `source/pipeline/step/`
- Retrieval query shaping: `source/pipeline/utils.py`
- Dense retriever: `source/module/retrieve/dense.py`
- Generator: `source/module/generate/llama.py`
- Indexing: `source/module/index/index.py`
- Prompts: `prompts/prompt_set__1/`
- Data/build scripts: `script/download/`
- Evaluation: `source/evaluation/`

## Working Rules

1. Giữ mọi thay đổi trong repo `DS319/ReSCORE` trừ khi người dùng yêu cầu rõ.
2. Tránh scan rộng hoặc sửa trong `data/`, `predictions/`, `__pycache__/` nếu
   task không thực sự cần.
3. Không xóa dataset, index, checkpoint, output hoặc file sinh ra số lượng lớn
   nếu chưa có xác nhận.
4. Khi mô tả pipeline, nói rõ đang bám theo `paper`, `repo active path` hay
   `ReSCORE_Pipeline_for_Reproduction.md`.
5. Khi đề xuất lệnh chạy, luôn ghi đủ `dataset`, `method`, `running_name` và
   các flag quan trọng.
6. Nếu test nặng GPU hoặc cần model ngoài, ưu tiên kiểm tra tĩnh hoặc smoke
   check trước và nói rõ giới hạn.
7. Logic trong `demo/` có thể khác pipeline training/inference chính; đừng coi
   demo là source of truth cho phương pháp nếu code chính nói khác.

## Reporting

Khi hoàn tất, nên báo ngắn gọn:

- đã đụng vào file nào trong repo này,
- có bám theo `ReSCORE_Pipeline_for_Reproduction.md` hay không,
- có cần GPU/model/data ngoài để xác minh đầy đủ hay không.
