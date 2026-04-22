Dưới đây là phần giải đáp chi tiết cho các câu hỏi của bạn về bài báo "ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision":

**1. Tại sao bài báo này được viết ra?** Bài báo được viết ra để giải quyết hạn chế của các hệ thống Hỏi đáp đa bước (Multi-hop Question Answering - MHQA). Trong MHQA, các mô hình truy xuất dày đặc (dense retrievers) thường vượt trội hơn các phương pháp thưa (sparse methods như BM25) nhưng lại cần các cặp truy vấn - tài liệu được gán nhãn để tinh chỉnh. Việc tạo ra dữ liệu gán nhãn này cho MHQA tốn rất nhiều công sức và chi phí do các truy vấn (các câu hỏi được viết lại) liên tục thay đổi qua mỗi bước lập luận.

**2. Bài toán cụ thể là gì?** Bài toán cụ thể là làm thế nào để huấn luyện một bộ truy xuất dày đặc (dense retriever) phục vụ cho hệ thống RAG lặp (iterative RAG) trong MHQA mà không cần phụ thuộc vào các tài liệu được con người gán nhãn về độ liên quan với các truy vấn.

**3. Giải pháp đề xuất là gì?** Các tác giả đề xuất **ReSCORE** (Retriever Supervision with Consistency and Relevance), một phương pháp mới sử dụng Mô hình ngôn ngữ lớn (LLM) để tạo ra các nhãn giả (pseudo-ground truth labels) nhằm huấn luyện mô hình truy xuất. Nhãn giả này được tính toán dựa trên xác suất mà LLM sinh ra cả câu hỏi ban đầu và câu trả lời đúng khi được cung cấp một tài liệu. Phương pháp này giúp mô hình hóa đồng thời cả **sự liên quan** (relevance) của tài liệu với câu hỏi và **sự nhất quán** (consistency) của tài liệu đối với câu trả lời. Bộ truy xuất được huấn luyện theo phương pháp này sau đó được tích hợp vào một hệ thống RAG lặp có tên là **IQATR**.

**4. Thực nghiệm được thiết kế thế nào?** Thực nghiệm được tiến hành trên 3 tập dữ liệu MHQA phổ biến: MuSiQue, 2WikiMHQA, và HotpotQA. Nhóm nghiên cứu sử dụng Llama-3.1-8B-Instruct kết hợp với mô hình truy xuất Contriever và đối chiếu với BM25. Hệ thống IQATR của họ được so sánh với các hệ thống SOTA (hiện đại nhất) về RAG lặp như ReAcT, FLARE, Self-RAG, Adaptive-Note, IRCoT, và Adaptive-RAG. Bên cạnh đó, họ cũng thực hiện các phân tích chuyên sâu (ablation studies) để đánh giá tác động của các loại nhãn giả khác nhau và các phương pháp viết lại truy vấn (query reformulation).

**5. Đánh giá bằng metric gì?** Bài báo sử dụng hai nhóm metric chính:

- **Đánh giá chất lượng trả lời (QA):** Sử dụng **Exact Match (EM)** và **F1 score** ở cấp độ câu trả lời theo tiêu chuẩn của các tập dữ liệu.
- **Đánh giá hiệu suất truy xuất lặp:** Giới thiệu một metric mới gọi là **multi-hop recall at k ($MHR@k$)**, giúp đo lường tỷ lệ các tài liệu gốc (ground truth) được truy xuất thành công cộng dồn qua từng bước lặp.

**6. Đóng góp thực sự là gì?** Đóng góp của bài báo gồm 3 điểm chính:

- Đề xuất ReSCORE, một cách tiếp cận mới để huấn luyện bộ truy xuất dày đặc lặp lại cho MHQA mà không cần dùng dữ liệu gán nhãn thủ công.
- Xây dựng hệ thống IQATR (sử dụng ReSCORE), đạt được hiệu suất cao nhất (SOTA) trên 3 bộ benchmark MHQA.
- Cung cấp phân tích sâu sắc về tác động của các nhãn pseudo-GT và các phương pháp viết lại truy vấn trong hệ thống RAG lặp.

**7. Hướng tương lai là gì?** Nhóm tác giả chỉ ra hai hạn chế cần giải quyết trong tương lai:

- **Khả năng tổng quát hóa:** Bộ truy xuất hiện được tinh chỉnh đặc thù cho các tập dữ liệu huấn luyện, do đó khả năng mở rộng (Out-of-Distribution) sang các tập dữ liệu có kiểu lập luận hoặc đặc điểm khác vẫn còn hạn chế.
- **Chi phí tính toán:** Việc sử dụng quy trình truy xuất lặp làm tăng độ trễ và chi phí tính toán (đặc biệt với các câu hỏi cần nhiều bước nhảy). Hướng đi tiếp theo là cần tối ưu hóa thêm để framework này hiệu quả và có khả năng nhân rộng trong thực tế.

**8. Liên quan đến kiến thức cũ của người đọc thế nào?** Công trình này được xây dựng dựa trên các khái niệm quen thuộc trong lĩnh vực xử lý ngôn ngữ tự nhiên hiện nay như **RAG (Retrieval-Augmented Generation)** và kỹ thuật **Iterative RAG** (thực hiện truy xuất lặp đi lặp lại để thu thập đủ thông tin cho các câu hỏi phức tạp). Nó cũng mở rộng kiến thức về cách tinh chỉnh mô hình tìm kiếm thông tin (như Contriever) bằng **Kullback-Leibler (KL) divergence loss**, biến LLM thành "giáo viên" để tạo ra nhãn huấn luyện (pseudo-GT) thay cho con người.

**9. Liên quan đến các công trình khác thế nào?**

- **Với các nghiên cứu huấn luyện Retriever cho RAG:** Các phương pháp cũ (như LLM-Embedder, ATLAS) thường chỉ tập trung vào câu hỏi đơn bước (single-hop) và tính nhất quán (consistency) mà bỏ qua MHQA và sự liên quan (relevance) của tài liệu.
- **Với các hệ thống Iterative RAG (như FLARE, Self-RAG, IRCoT, Adaptive-RAG):** Những công trình này chỉ tập trung vào cách RAG động (dynamic retrieval) nhưng lại phụ thuộc vào các bộ truy xuất cũ (như BM25) hoặc các bộ truy xuất được huấn luyện trên domain khác. Bài báo này khác biệt ở chỗ **trực tiếp huấn luyện bộ truy xuất ngay bên trong hệ thống RAG lặp**.
- **Với các nghiên cứu về Giám sát bằng LLM (LLM Supervision):** Đây là nghiên cứu đầu tiên tận dụng LLM để huấn luyện một bộ truy xuất cho cấu trúc RAG lặp dành riêng cho bài toán MHQA.

---

Phần Tổng quan các công trình liên quan (Related Work) của bài báo được chia thành 3 nhóm nghiên cứu chính, giúp làm nổi bật những điểm mà các công trình trước đây chưa giải quyết được:

**1. Huấn luyện bộ truy xuất cho RAG (Training Retrievers for RAG)**

- **Cách làm trước đây:** Nhiều phương pháp tập trung vào việc nâng cao chất lượng tìm kiếm tài liệu thông qua học có giám sát bằng các bộ dữ liệu lớn được gán nhãn, hoặc học không giám sát. Để thu hẹp khoảng cách giữa quá trình tìm kiếm và quá trình sinh văn bản, các công trình như LLM-Embedder, Intermediate Distillation, REPLUG và ATLAS đã sử dụng Mô hình ngôn ngữ lớn (LLM) để định hướng cho bộ truy xuất.
- **Hạn chế:** Các phương pháp này hầu hết chỉ tập trung vào các câu hỏi đơn bước (single-hop) và chỉ xem xét sự nhất quán (consistency) của tài liệu với câu trả lời. Chúng hoàn toàn bỏ qua quá trình lập luận lặp (iterative reasoning), bỏ qua bài toán Hỏi đáp đa bước (MHQA) cũng như tính liên quan (relevance) của tài liệu.

**2. Quá trình RAG lặp (Iterative RAG)**

- **Cách làm trước đây:** Đây là phương pháp mở rộng từ RAG đơn bước để giải quyết các truy vấn phức tạp cần tổng hợp từ nhiều tài liệu. Một số công trình tiêu biểu bao gồm:
    - **FLARE:** Tìm kiếm tài liệu một cách thích ứng khi mô hình sinh ra các token có xác suất thấp.
    - **Self-RAG:** Tự học cách phân loại để quyết định khi nào cần gọi bộ truy xuất bên ngoài.
    - **ITER-RETGEN:** Sử dụng luôn kết quả đầu ra của bước trước đó để làm ngữ cảnh truy xuất tiếp theo.
    - **IRCoT:** Kết hợp chuỗi suy luận (Chain of Thoughts) lặp đi lặp lại để bắt chước quy trình lập luận đa bước của con người.
    - **Adaptive-RAG** (mở rộng từ IRCoT) và **Adaptive-Note**: Cải thiện tính hiệu quả bằng cách tự điều chỉnh số bước lập luận dựa trên độ phức tạp của câu hỏi, hoặc dùng LLM để lọc bớt tài liệu thừa.
- **Hạn chế:** Không có công trình nào trong số này tập trung vào việc huấn luyện chính bộ truy xuất. Chúng thường dựa vào các phương pháp cũ như BM25 hoặc sử dụng các mô hình tìm kiếm được huấn luyện sẵn trên các tập dữ liệu khác. Ngược lại, ReSCORE tiến hành huấn luyện trực tiếp bộ truy xuất dày đặc (dense retriever) ngay bên trong vòng lặp RAG.

**3. Huấn luyện với sự giám sát của LLM (Training with LLM Supervision)**

- **Cách làm trước đây:** Sử dụng các LLM lớn đóng vai trò làm "giáo viên" để tạo ra dữ liệu huấn luyện cho các mô hình nhỏ hơn, nhằm khắc phục việc thiếu hụt dữ liệu gán nhãn bởi con người.
    - Ví dụ như CoT-Distill (dùng chuỗi suy luận của LLM), Self-RAG (dùng dữ liệu tạo bởi GPT-4).
    - Hoặc dùng danh sách tài liệu được xếp hạng bởi LLM để hướng dẫn huấn luyện (như Intermediate Distillation, Promptagator, RankVicuna).
    - Mô hình ATLAS sử dụng xác suất dự đoán token từ mô hình giáo viên để huấn luyện bộ truy xuất.
- **Sự khác biệt của bài báo:** Nhóm tác giả khẳng định ReSCORE là nghiên cứu **đầu tiên** tận dụng xác suất từ LLM để trực tiếp huấn luyện bộ truy xuất nằm gọn trong một kiến trúc RAG lặp để phục vụ cho bài toán MHQA.

---
Phần 3 (Methods) của bài báo mô tả chi tiết framework cốt lõi, được chia thành hai phần chính: **Kiến trúc Iterative RAG (IQATR)** dùng để suy luận tìm câu trả lời, và **Phương pháp ReSCORE** dùng để huấn luyện bộ truy xuất (retriever). Dưới đây là mô tả chi tiết từng thành phần để bạn có thể nắm bắt và reproduce (tái tạo) lại framework này.

### 3.1. Iterative RAG Framework (Kiến trúc RAG lặp)

Mục tiêu của phần này là giải quyết câu hỏi phức tạp (multi-hop) $q$ bằng cách truy xuất lặp đi lặp lại để thu thập đủ một tập hợp các tài liệu liên quan $D^*$. Quá trình này diễn ra theo các bước (iterations) như sau:

**Tại vòng lặp đầu tiên ($i = 1$):**

1. **Truy xuất ban đầu:** Lấy câu hỏi gốc làm truy vấn $q^{(1)} = q$. Dùng bộ truy xuất để lấy ra top $k$ tài liệu, gọi là tập $D^{(1)}$.
2. **Dự đoán câu trả lời:** Đưa $D^{(1)}$ vào một LLM thông qua prompt. LLM sẽ quyết định một trong hai hướng:
    - Nếu thông tin trong $D^{(1)}$ đã đủ: LLM sinh ra câu trả lời cuối cùng $a^{(1)}$ và quá trình lặp kết thúc.
    - Nếu thiếu thông tin: LLM sinh ra chuỗi "unknown" (không biết), báo hiệu cần tìm kiếm thêm.
3. **Sinh "Thought" (Suy nghĩ/Tóm tắt):** Nếu LLM trả về "unknown", hệ thống yêu cầu LLM viết một "thought" $t^{(1)}$. Đây là một câu duy nhất chắt lọc các thông tin quan trọng nhất từ $D^{(1)}$ có ích cho câu hỏi gốc. Việc này giúp nén thông tin để không bị vượt quá giới hạn ngữ cảnh (context limit) ở các bước sau.
4. **Viết lại truy vấn (Query Reformulation):** Cuối cùng, LLM tạo ra một truy vấn mới $q^{(2)}$, tập trung vào những khía cạnh còn thiếu hoặc chưa được giải quyết của $q^{(1)}$ để dùng cho vòng lặp tiếp theo.

**Tại các vòng lặp tiếp theo ($i > 1$):**

1. **Truy xuất tiếp nối:** Dùng truy vấn mới $q^{(i)}$ để lấy thêm $k$ tài liệu mới $D^{(i)}$ (những tài liệu này không được trùng với các tài liệu đã lấy ở bước trước).
2. **Tổng hợp và Dự đoán:** LLM được cung cấp các tài liệu mới $D^{(i)}$ cùng với **tất cả các thoughts từ các bước trước** ($t^{(1)}, ..., t^{(i-1)}$). Nó lại tiếp tục đánh giá xem đã đủ thông tin trả lời chưa.
3. Quá trình này lặp lại (sinh $t^{(i)}$ và $q^{(i+1)}$) cho đến khi LLM đưa ra được câu trả lời cuối cùng (khác "unknown") hoặc đạt đến giới hạn số vòng lặp tối đa.

### 3.2. Training Retriever for Iterative RAG (Phương pháp ReSCORE)

Đây là phần cốt lõi nhất hướng dẫn cách huấn luyện bộ truy xuất (dense retriever) mà **không cần nhãn dữ liệu thủ công**. Bài báo dùng chính LLM để tạo ra "nhãn giả" (pseudo-Ground Truth - pseudo-GT).

**1. Cách tạo nhãn Pseudo-GT bằng LLM:** Thay vì dùng con người đánh giá xem tài liệu $d^{(i)}_j$ có liên quan đến truy vấn $q^{(i)}$ hay không, ReSCORE tính toán xác suất phân bố $Q^{(i)}_{LM}$ dựa trên LLM.

- Trực giác của bài báo: Một tài liệu quan trọng là tài liệu mà khi LLM đọc nó, LLM có thể dễ dàng sinh ra **cả câu hỏi ban đầu ($q$) và câu trả lời đúng ($a$)**.
- Công thức phân bố nhãn giả: $Q^{(i)}_{LM}(d^{(i)}_j | q) \propto P^{(i)}_{LM}(a, q | d^{(i)}_j)$.
- Theo quy tắc chuỗi (chain rule), công thức này được phân rã thành 2 thành phần: $$P^{(i)}_{LM}(q | d^{(i)}_j) \cdot P^{(i)}_{LM}(a | q, d^{(i)}_j)$$
    - **Sự liên quan (Relevance):** Thành phần $P^{(i)}_{LM}(q | d^{(i)}_j)$ là xác suất sinh ra câu hỏi từ tài liệu. Nó giúp loại bỏ các tài liệu không đúng chủ đề.
    - **Sự nhất quán (Consistency):** Thành phần $P^{(i)}_{LM}(a | q, d^{(i)}_j)$ là xác suất trả lời đúng khi có tài liệu. Nó giúp đảm bảo tài liệu thực sự chứa thông tin giải quyết câu hỏi, tránh trường hợp tài liệu chỉ chứa các từ khóa trùng lặp bề ngoài.

**2. Hàm Loss (Hàm mất mát) để huấn luyện:** Bộ truy xuất (Retriever) được huấn luyện để phân bố dự đoán của nó ($P^{(i)}_R$) tiệm cận với phân bố nhãn giả của LLM ($Q^{(i)}_{LM}$).

- Hệ thống tối ưu hóa bằng cách cực tiểu hóa độ phân kỳ Kullback-Leibler (KL Divergence loss): $$\sum_{n=1}^N \sum_{i=0}^{\eta_n} D_{KL} \left( Q^{(i)}_{LM}(D^{(i)} | q^{(i)}_n) \parallel P^{(i)}_R(D^{(i)} | q^{(i)}_n) \right)$$ (Trong đó $N$ là số lượng cặp QA, $\eta_n$ là số vòng lặp của câu hỏi đó).
- Phân bố của bộ truy xuất $P^{(i)}_R$ được tính bằng hàm Softmax trên tích vô hướng (dot product) giữa vector nhúng của câu hỏi (query embedding) và vector nhúng của tài liệu (document embedding).

**3. Tối ưu hóa tính toán (Chi tiết cực kỳ quan trọng để reproduce):** Tính toán phân bố $Q^{(i)}_{LM}$ trên toàn bộ cơ sở dữ liệu (hàng triệu tài liệu) bằng LLM là bất khả thi vì chi phí tính toán quá đắt. Do đó, tại mỗi vòng lặp $i$, hệ thống **chỉ lấy mẫu top $M$ tài liệu** có điểm số cao nhất từ bộ truy xuất hiện tại (bài báo chọn $M = 32$) và chỉ cho LLM tính $Q^{(i)}_{LM}$ trên tập nhỏ $M$ tài liệu này để làm chuẩn (normalize và tính loss).

### Tóm tắt các setup quan trọng để Reproduce (từ phần Thực nghiệm):

Để code lại framework này, bạn cần chú ý các thông số kỹ thuật (Implementation Details) được nhắc đến ở mục 4.1:

- **Bộ nhúng (Embedder):** Chỉ huấn luyện bộ mã hóa câu hỏi (Question Embedder), đóng băng (freeze) bộ mã hóa tài liệu (Document Embedder) trong suốt quá trình.
- **Tham số số lượng:** Khi huấn luyện (loss calculation) lấy top $M = 32$ tài liệu, nhưng khi suy luận thực tế (inference) chỉ lấy top $k = 8$ tài liệu mỗi vòng lặp.
- **Giới hạn lặp:** Số vòng lặp tối đa $\eta_n = 6$. MHQA yêu cầu tối thiểu 2 bước nhảy, nên số vòng lặp tối thiểu là 2.
- Nhiệt độ (Temperature) của LLM khi sinh output được set ở $0.1$ để đảm bảo tính ổn định.