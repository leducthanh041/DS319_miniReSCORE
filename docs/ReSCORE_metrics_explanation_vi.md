# Giải thích các độ đo dùng để báo cáo ReSCORE/TTA

Tài liệu này giải thích các độ đo đang được dùng trong repo ReSCORE hiện tại khi báo cáo kết quả baseline, ReSCORE checkpoint và Test-Time Adaptation (TTA). Nội dung bám theo `docs/ReSCORE-paper.pdf` và execution path hiện tại trong `source/`.

## 1. Nhóm độ đo QA

Trong paper ReSCORE, phần đánh giá QA chính dùng hai độ đo chuẩn cho Multi-hop Question Answering:

- `EM`
- `F1`

Trong repo hiện tại, file `test_evaluation.json` còn ghi thêm:

- `precision`
- `recall`
- `count`
- `sp_em`, `sp_f1`, `sp_precision`, `sp_recall`

### EM

`EM` là viết tắt của Exact Match.

Ý nghĩa: tỷ lệ câu hỏi mà câu trả lời dự đoán khớp hoàn toàn với đáp án chuẩn sau bước chuẩn hóa.

Ví dụ:

- Đáp án chuẩn: `Barack Obama`
- Dự đoán: `Barack Obama`
- Kết quả: đúng Exact Match

Nếu dự đoán chỉ gần đúng, thiếu từ, thừa từ, hoặc diễn đạt khác nhưng không khớp theo evaluator thì `EM = 0` cho câu đó.

Cách đọc khi báo cáo:

- `EM` càng cao càng tốt.
- `EM` là metric nghiêm ngặt nhất ở tầng câu trả lời.
- Nếu TTA làm `F1` tăng nhưng `EM` gần như không tăng, điều đó có nghĩa là mô hình trả lời gần đúng hơn nhưng chưa đủ chính xác để khớp hoàn toàn.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| EM | 9.6% | 10.2% | +0.6 điểm |

Diễn giải: TTA có cải thiện EM nhưng mức tăng còn nhỏ.

### F1

`F1` đo mức độ chồng khớp token giữa câu trả lời dự đoán và đáp án chuẩn.

Ý nghĩa: nếu câu trả lời chưa khớp hoàn toàn nhưng chứa nhiều token đúng, F1 vẫn có thể tăng.

F1 được tính từ precision và recall ở cấp token:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Cách đọc khi báo cáo:

- `F1` càng cao càng tốt.
- `F1` mềm hơn `EM`, phù hợp để quan sát cải thiện từng phần trong câu trả lời.
- Khi `F1` tăng nhưng `EM` không tăng nhiều, mô hình đang tiến gần hơn đến đáp án đúng nhưng vẫn còn lỗi format, thiếu/thừa token hoặc suy luận chưa đủ chính xác.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| F1 | 17.7% | 20.3% | +2.6 điểm |

Diễn giải: TTA cải thiện chất lượng trả lời ở mức token khá rõ.

### Precision

Trong `test_evaluation.json`, `precision` là precision ở cấp token của câu trả lời, không phải `Precision@8` retrieval.

Ý nghĩa: trong các token mà mô hình dự đoán, tỷ lệ token khớp với đáp án chuẩn là bao nhiêu.

Ví dụ trực quan:

- Dự đoán dài và chứa nhiều token thừa thì precision có thể thấp.
- Dự đoán ngắn, ít thừa token và chứa token đúng thì precision cao hơn.

Cách đọc khi báo cáo:

- `precision` càng cao càng tốt.
- Precision thấp thường cho thấy mô hình trả lời quá dài, lan man hoặc thêm thông tin không cần thiết.
- Nếu precision tăng, câu trả lời của mô hình trở nên gọn và đúng trọng tâm hơn.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| Precision | 19.2% | 21.2% | +2.0 điểm |

Diễn giải: TTA giúp câu trả lời bớt nhiễu hơn hoặc chứa tỷ lệ token đúng cao hơn.

### Recall

Trong `test_evaluation.json`, `recall` là recall ở cấp token của câu trả lời, không phải `Recall@8` retrieval.

Ý nghĩa: trong các token thuộc đáp án chuẩn, mô hình dự đoán được bao nhiêu token.

Cách đọc khi báo cáo:

- `recall` càng cao càng tốt.
- Recall thấp thường cho thấy mô hình bỏ sót một phần đáp án.
- Nếu recall tăng mạnh hơn precision, mô hình đang lấy được nhiều thông tin đúng hơn nhưng có thể vẫn còn token thừa.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| Recall | 20.9% | 25.9% | +5.0 điểm |

Diễn giải: TTA giúp mô hình bao phủ nhiều token đúng hơn trong đáp án.

### Count

`count` là số lượng câu hỏi được đưa vào đánh giá.

Ý nghĩa: đây không phải metric chất lượng, mà là điều kiện để so sánh công bằng.

Cách đọc khi báo cáo:

- `count` của baseline và TTA phải giống nhau.
- Nếu `count` khác nhau, không nên so sánh trực tiếp EM/F1/Precision/Recall vì hai hệ thống đang được đánh giá trên tập mẫu khác nhau.

Trong kết quả hiện tại của bạn:

| Setting | Count |
|---|---:|
| Baseline fair | 500 |
| TTA | 500 |

Diễn giải: so sánh hiện tại là hợp lệ về số lượng mẫu.

### SP metrics

Các trường `sp_em`, `sp_f1`, `sp_precision`, `sp_recall` xuất hiện trong output hiện tại. Tên `sp` thường liên quan đến supporting facts/evidence trong một số evaluator MHQA.

Tuy nhiên, trong execution path hiện tại của repo, các giá trị này đang trùng hoặc gần trùng với answer metric trong một số output. Vì vậy khi báo cáo chính, nên ưu tiên:

- `EM`
- `F1`
- `precision`
- `recall`
- `MHR@8`

Chỉ báo cáo `sp_*` nếu bạn kiểm tra chắc evaluator dataset đang tính supporting fact đúng như official evaluator.

## 2. Nhóm độ đo Retrieval

Paper ReSCORE giới thiệu metric chính cho retrieval là `MHR@k`, tức Multi-Hop Recall at k.

Trong paper, công thức tổng quát là:

```text
MHR_i@k = |D* ∩ union_{l=1..i} D^(l)| / |D*|
```

Trong đó:

- `D*` là tập tài liệu hỗ trợ chuẩn.
- `D^(l)` là tập tài liệu được truy xuất ở iteration/hop thứ `l`.
- `i` là số iteration đã xét.
- `k` là số tài liệu lấy ra mỗi lần retrieval.

Paper dùng `k = 8` khi inference, nên độ đo chính là `MHR_i@8`.

### MHR@8

`MHR@8` đo tỷ lệ tài liệu hỗ trợ chuẩn đã được tìm thấy trong quá trình retrieval lặp, với top 8 tài liệu mỗi lần retrieval.

Ý nghĩa: đây là metric chính để đánh giá retrieval trong iterative multi-hop QA.

Khác với QA F1/EM, `MHR@8` không đánh giá câu trả lời cuối cùng. Nó đánh giá hệ thống có truy xuất được đủ bằng chứng cần thiết hay không.

Cách đọc:

- `MHR_1@8`: recall sau iteration đầu tiên.
- `MHR_2@8`: recall tích lũy sau hai iteration.
- `MHR_final@8`: recall tích lũy sau toàn bộ quá trình retrieval/generation.

Nếu `MHR_i@8` tăng khi `i` tăng, điều đó cho thấy hệ thống retrieval lặp đang tìm thêm được bằng chứng mới qua các hop sau.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| MHR_1@8 | 25.22% | 30.53% | +5.32 điểm |
| MHR_2@8 | 26.17% | 34.17% | +8.00 điểm |
| MHR_3@8 | 26.22% | 34.63% | +8.42 điểm |
| MHR_final@8 | 26.22% | 34.90% | +8.68 điểm |

Diễn giải: TTA cải thiện retrieval rõ rệt, đặc biệt từ hop 2 trở đi. Điều này phù hợp với mục tiêu của iterative retrieval: truy xuất thêm bằng chứng bổ sung qua từng bước suy luận.

### MHR@8 title-only

`title_only_MHR@8` là biến thể trong repo hiện tại. Nó đánh giá retrieval dựa trên việc title của document có khớp với gold supporting document hay không.

Ý nghĩa: đây là metric nới lỏng hơn so với matching đầy đủ title + paragraph/content.

Vì sao cần title-only:

- Một số dataset hoặc database có thể chia passage theo đoạn khác nhau.
- Cùng một Wikipedia page/title có thể xuất hiện ở nhiều đoạn.
- Title-only giúp đánh giá liệu hệ thống có tìm đúng thực thể/trang liên quan hay không, ngay cả khi đoạn cụ thể chưa khớp hoàn toàn.

Cách đọc:

- `title_only_MHR@8` thường cao hơn hoặc bằng `MHR@8`.
- Nếu `title_only_MHR@8` cao hơn nhiều so với `MHR@8`, hệ thống đang tìm đúng page/entity nhưng chưa tìm đúng passage cụ thể.

Trong kết quả hiện tại của bạn trên MuSiQue InD:

| Metric | Baseline fair | TTA | Cải thiện |
|---|---:|---:|---:|
| title-only MHR_final@8 | 31.38% | 39.33% | +7.95 điểm |

Diễn giải: TTA giúp truy xuất đúng title/entity tốt hơn rõ rệt. Khoảng cách giữa title-only MHR và MHR đầy đủ cho thấy vẫn còn lỗi ở mức chọn đúng đoạn passage.

## 3. Precision@8 và Recall@8

Trong paper ReSCORE, metric retrieval chính được báo cáo là `MHR_i@8`, không phải Precision@8/Recall@8 truyền thống.

Nếu bạn muốn báo cáo thêm Precision@8/Recall@8, cần phân biệt rõ:

### Precision@8 retrieval

Ý nghĩa: trong 8 tài liệu được truy xuất, có bao nhiêu tài liệu là relevant/gold evidence.

```text
Precision@8 = số tài liệu đúng trong top 8 / 8
```

Metric này trả lời câu hỏi: top 8 có sạch không, có ít nhiễu không?

### Recall@8 retrieval

Ý nghĩa: trong toàn bộ tài liệu hỗ trợ chuẩn, top 8 tìm được bao nhiêu tài liệu.

```text
Recall@8 = số tài liệu hỗ trợ chuẩn tìm được trong top 8 / tổng số tài liệu hỗ trợ chuẩn
```

Metric này trả lời câu hỏi: top 8 có bao phủ đủ evidence không?

### Khác biệt với precision/recall trong `test_evaluation.json`

Rất quan trọng:

- `precision` trong `test_evaluation.json` là answer-token precision.
- `recall` trong `test_evaluation.json` là answer-token recall.
- `Precision@8` và `Recall@8` là retrieval metrics, không phải answer metrics.

Do đó khi báo cáo, nên ghi rõ tên:

- `Answer Precision`
- `Answer Recall`
- `Retrieval Precision@8`
- `Retrieval Recall@8`
- `MHR@8`

Nếu repo hiện tại chưa xuất trực tiếp `Precision@8`/`Recall@8`, không nên gọi answer precision/recall là Precision@8/Recall@8.

## 4. Cách diễn giải kết quả TTA

Khi so sánh baseline fair và TTA, nên đọc theo hai tầng:

### Tầng retrieval

Các metric chính:

- `MHR_1@8`
- `MHR_2@8`
- `MHR_final@8`
- `title_only_MHR_final@8`

Nếu các metric này tăng, TTA đang giúp retriever thích nghi tốt hơn với từng câu hỏi test.

Trong kết quả InD hiện tại, `MHR_final@8` tăng `+8.68` điểm. Đây là cải thiện mạnh ở tầng retrieval.

### Tầng QA

Các metric chính:

- `EM`
- `F1`
- `Answer Precision`
- `Answer Recall`

Nếu QA metric tăng, hệ thống không chỉ truy xuất tốt hơn mà còn chuyển được evidence thành câu trả lời tốt hơn.

Trong kết quả InD hiện tại:

- `F1` tăng `+2.6` điểm.
- `Answer Recall` tăng `+5.0` điểm.
- `EM` chỉ tăng `+0.6` điểm.

Diễn giải: TTA giúp câu trả lời chứa nhiều thông tin đúng hơn, nhưng chưa đủ để tăng mạnh exact match.

## 5. Ngưỡng cải thiện nên xem là có ý nghĩa

Không có ngưỡng tuyệt đối đúng cho mọi thí nghiệm. Tuy nhiên với quy mô hiện tại `count = 500`, có thể dùng ngưỡng thực nghiệm sau để diễn giải:

| Metric | Cải thiện đáng chú ý | Cải thiện mạnh |
|---|---:|---:|
| EM | +0.5 đến +1.0 điểm | >= +2.0 điểm |
| F1 | +1.5 đến +2.0 điểm | >= +3.0 điểm |
| Answer Precision | +1.0 đến +2.0 điểm | >= +3.0 điểm |
| Answer Recall | +2.0 đến +3.0 điểm | >= +4.0 điểm |
| MHR@8 | +4.0 đến +5.0 điểm | >= +7.0 điểm |
| MHR@8 title-only | +4.0 đến +5.0 điểm | >= +7.0 điểm |

Với OOD setting, kỳ vọng hợp lý là:

- QA metric có thể tăng ít hơn InD.
- Retrieval metric, đặc biệt `MHR@8`, nên tăng rõ nếu TTA thật sự giúp thích nghi domain.
- Nếu `MHR@8` tăng nhưng `EM/F1` không tăng tương ứng, bottleneck có thể nằm ở generator hoặc prompt, không phải retriever.

## 6. Gợi ý câu chữ khi báo cáo

Có thể viết:

> Following ReSCORE, we report answer-level EM and F1 for MHQA quality, and MHR@8 for iterative retrieval quality. In addition, we report answer-level precision/recall produced by the local evaluator and a title-only variant of MHR@8 to diagnose whether retrieval reaches the correct entity/page even when the exact passage does not match.

Diễn giải tiếng Việt:

> Theo ReSCORE, chúng tôi báo cáo EM và F1 ở cấp câu trả lời để đánh giá chất lượng QA, và MHR@8 để đánh giá khả năng truy xuất bằng chứng trong pipeline retrieval lặp. Ngoài ra, chúng tôi báo cáo precision/recall ở cấp token của câu trả lời từ evaluator hiện tại, cùng với biến thể title-only MHR@8 để phân tích liệu hệ thống đã truy xuất đúng thực thể/trang liên quan hay chưa.

Khi nói về TTA:

> TTA cải thiện rõ rệt MHR@8, cho thấy quá trình thích nghi tại test-time giúp truy xuất nhiều bằng chứng hỗ trợ hơn qua các hop. Mức cải thiện QA nhỏ hơn retrieval, cho thấy generator vẫn là bottleneck trong việc chuyển evidence đúng thành câu trả lời exact-match.

## 7. Metric nên đưa vào bảng chính

Bảng chính nên gồm:

| Nhóm | Metric |
|---|---|
| QA | EM |
| QA | F1 |
| QA phụ trợ | Answer Precision |
| QA phụ trợ | Answer Recall |
| Retrieval | MHR_1@8 |
| Retrieval | MHR_2@8 |
| Retrieval | MHR_final@8 |
| Retrieval phụ trợ | title-only MHR_final@8 |
| Sanity check | Count |

Nếu bảng quá rộng, ưu tiên:

- `EM`
- `F1`
- `MHR_final@8`
- `title-only MHR_final@8`
- `count`

