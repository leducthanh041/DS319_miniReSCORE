# Test-Time Adaptation for ReSCORE-based Iterative Retrieval

**Loại tài liệu:** Research Design Document  
**Paper nền tảng:** ReSCORE (ACL 2025), TOUR (ACL 2023 Findings)  
**Mục tiêu:** Cải thiện OOD generalization của ReSCORE mà không retraining toàn bộ retriever  
**Phân loại nguồn gốc ý tưởng:** `[FACT]` = có trong paper · `[INFERENCE]` = suy luận hợp lý · `[HYPOTHESIS]` = chưa kiểm chứng

---

## 0. Bảng ký hiệu toán học

Phần này định nghĩa tất cả ký hiệu được dùng trong tài liệu. Mỗi ký hiệu được chú thích rõ nguồn gốc.

### 0.1. Ký hiệu dùng chung (cả TOUR và ReSCORE)

| Ký hiệu | Tên đầy đủ | Ý nghĩa | Nguồn |
|---|---|---|---|
| $q$ | Query | Câu hỏi / câu truy vấn gốc | Cả hai paper |
| $d_j$ | Document $j$ | Tài liệu thứ $j$ trong corpus | Cả hai paper |
| $D$ | Document corpus | Toàn bộ cơ sở dữ liệu tài liệu | Cả hai paper |
| $k$ | Top-$k$ | Số tài liệu được retrieve tại mỗi vòng | Cả hai paper |
| $M$ | Candidate pool | Số tài liệu được xét khi tính pseudo-GT ($M \gg k$, ReSCORE dùng $M=32$) | ReSCORE |
| $\tau$ | Temperature | Tham số nhiệt độ cho softmax — $\tau$ nhỏ → phân phối sắc nét hơn | TOUR |
| $\eta$ | Learning rate | Tốc độ học của gradient descent | TOUR |
| $D_{KL}(P \| Q)$ | KL Divergence | Độ phân kỳ Kullback–Leibler từ $Q$ đến $P$; đo mức độ khác nhau giữa hai phân phối | Cả hai paper |

### 0.2. Ký hiệu đặc thù của TOUR (single-hop retrieval)

| Ký hiệu | Tên đầy đủ | Ý nghĩa |
|---|---|---|
| $c$ | Context | Tài liệu / đoạn văn được retrieve (TOUR dùng "context", ReSCORE dùng "document" — cùng ý nghĩa) |
| $c^*$ | Gold context | Tài liệu **đúng** (ground-truth), chứa câu trả lời chính xác. Chỉ biết trong training, **không biết** tại test time |
| $\tilde{c}$ | Pseudo-positive context | Tài liệu được cross-encoder **đánh giá là relevant** và chọn làm positive tạm thời — đây là **nhãn giả**, đóng vai trò thay thế $c^*$ tại test time |
| $C^q_{1:k}$ | Top-$k$ retrieved contexts | Danh sách $k$ tài liệu được retrieve tốt nhất cho query $q$, sắp xếp giảm dần theo similarity score |
| $C^q_{hard}$ | Hard pseudo-positive set | Tập con của $C^q_{1:k}$, gồm các $\tilde{c}$ mà cross-encoder đánh giá tổng xác suất $\geq p$ (nucleus-style selection) |
| $\phi(q, c)$ | Cross-encoder score | Điểm số của cross-encoder $\phi$ đánh giá mức độ relevant của tài liệu $c$ với query $q$. Cross-encoder đọc đồng thời cả $q$ và $c$ (khác dual encoder) |
| $\text{sim}(q, c)$ | Similarity score | Độ tương đồng giữa query và document: $\text{sim}(q,c) = E_q(q)^\top E_c(c)$ — tích vô hướng của hai vector nhúng |
| $P_k(c \mid q)$ | Retriever distribution | Phân phối xác suất của retriever trên top-$k$ documents: $P_k(c_i \mid q) = \dfrac{\exp(\text{sim}(q, c_i))}{\sum_{j=1}^{k} \exp(\text{sim}(q, c_j))}$ |
| $P(c \mid q, \phi)$ | Cross-encoder distribution | Phân phối xác suất từ cross-encoder (soft pseudo-label): $P(c_i \mid q, \phi) = \dfrac{\exp(\phi(q, c_i)/\tau)}{\sum_{j=1}^{k} \exp(\phi(q, c_j)/\tau)}$ |
| $q_t$ | Query vector tại step $t$ | Vector biểu diễn của câu truy vấn sau $t$ bước gradient descent. $q_0 = E_q(q)$ là điểm khởi đầu từ encoder |
| $p$ | Nucleus threshold | Ngưỡng tích lũy xác suất để chọn $C^q_{hard}$ (TOUR dùng $p=0.5$); tương tự nucleus sampling trong sinh ngôn ngữ |
| $g(q_t, C^{q_t}_{1:k})$ | Update function | Hàm cập nhật query vector theo tinh thần Rocchio algorithm |

> **Lưu ý về $\tilde{c}$ vs $c^*$:** Đây là điểm mấu chốt của TOUR. Tại training, người ta có $c^*$ (tài liệu đúng thực sự). Tại test time, không biết $c^*$, nên TOUR dùng cross-encoder để chọn ra $\tilde{c}$ làm **proxy** cho $c^*$. $\tilde{c}$ là **nhãn giả** (pseudo-label) — có thể sai, nhưng thường đủ tốt để cải thiện retrieval.

### 0.3. Ký hiệu đặc thù của ReSCORE (iterative multi-hop)

| Ký hiệu | Tên đầy đủ | Ý nghĩa |
|---|---|---|
| $i$ | Iteration index | Chỉ số vòng lặp trong iterative retrieval ($i = 1, 2, \ldots, \eta_n$) |
| $q^{(i)}$ | Query tại iteration $i$ | Câu truy vấn tại vòng lặp $i$ (có thể là câu hỏi gốc $q$ hoặc đã được reformulate) |
| $D^{(i)}$ | Documents tại iteration $i$ | Tập $k$ tài liệu được retrieve tại vòng $i$; $D^{(i)} \cap D^{(j)} = \emptyset$ với $i \neq j$ (không lấy trùng) |
| $a$ | Answer | Câu trả lời đúng của question. **Có trong training, không có tại test time** |
| $t^{(i)}$ | Thought | Một câu tóm tắt thông tin quan trọng từ $D^{(i)}$, được LLM sinh ra để dùng cho iteration tiếp theo |
| $\eta_n$ | Max iterations | Số vòng lặp tối đa cho question $n$ (LLM quyết định động; ReSCORE giới hạn $\eta_n \leq 6$) |
| $N$ | Training set size | Số lượng cặp QA trong training set |
| $Q^{(i)}_{LM}$ | LLM pseudo-GT distribution | Phân phối xác suất pseudo-GT do LLM tính tại iteration $i$; đây là "nhãn giáo viên" trong ReSCORE |
| $P^{(i)}_R$ | Retriever distribution | Phân phối xác suất của retriever tại iteration $i$; đây là "output học sinh" mà retriever cần học |
| $P_{LM}$ | LLM probability | Xác suất mà LLM gán cho một chuỗi token. Ví dụ: $P_{LM}(q \mid d)$ = xác suất LLM sinh ra câu $q$ nếu cho trước document $d$ |
| $E_q(\cdot)$ | Query encoder | Hàm mã hóa câu truy vấn thành vector. Chỉ $E_q$ được cập nhật (train); $E_c$ bị đóng băng |
| $E_c(\cdot)$ | Document encoder | Hàm mã hóa tài liệu thành vector. Bị **đóng băng** (frozen) trong suốt training và test |

### 0.4. Ký hiệu của framework đề xuất (ReSCORE-TTA)

| Ký hiệu | Tên đầy đủ | Ý nghĩa |
|---|---|---|
| $\tilde{Q}^{(i)}_{TTA}$ | TTA pseudo-GT | Pseudo-GT distribution tại test time, thay thế $Q^{(i)}_{LM}$ vì không có $a$ |
| $q^{(i)}_t$ | Query vector tại step $t$, iteration $i$ | Vector của query $q^{(i)}$ sau $t$ bước Level 1 optimization |
| $q^{(i)}_0$ | Initial query vector | $E_q(q^{(i)})$ với tham số hiện tại (sau khi LoRA áp dụng) |
| $\theta_{Eq}$ | Query encoder parameters | Tham số gốc của query encoder sau khi train ReSCORE |
| $\Delta W = BA$ | LoRA update | Ma trận cập nhật low-rank; $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, rank $r \ll d$ |
| $P^{(i),\text{LoRA}}_R$ | LoRA-adapted retriever distribution | Phân phối retriever sau khi áp dụng LoRA |
| $\alpha, \beta, \gamma$ | Loss weights | Hệ số cân bằng giữa các thành phần trong $\mathcal{L}_{TTA}$ |
| $\lambda_1, \lambda_2$ | Regularization weights | Hệ số cho anchor regularization và LoRA norm bound |
| $\theta_{conf}$ | Confidence threshold | Ngưỡng để lọc pseudo-labels không đáng tin |
| $T_{inner}$ | Inner optimization steps | Số bước gradient descent trong Level 1 (TOUR), mặc định $\leq 3$ |

---

## 1. Tóm tắt điều hành

ReSCORE (ACL 2025) huấn luyện một dense retriever cho iterative MHQA bằng pseudo-GT labels từ LLM, đạt SOTA trên MuSiQue / HotpotQA / 2WikiMHQA. Tuy nhiên, các tác giả **tự thừa nhận hạn chế OOD** `[FACT]`: *"its ability to generalize to other datasets that differ in reasoning patterns or dataset characteristics remains limited."*

TOUR (ACL 2023 Findings) đề xuất tối ưu hóa **vector biểu diễn câu truy vấn** $q_t$ tại test time thông qua gradient descent, có hướng dẫn từ cross-encoder reranker $\phi(q,c)$, hoàn toàn không cập nhật tham số encoder `[FACT]`.

Tài liệu này đề xuất **ReSCORE-TTA**: một framework hai cấp độ `[HYPOTHESIS]`:

- **Cấp 1** – *Query Vector Optimization* (mở rộng TOUR sang iterative setting): tối ưu hóa vector $q^{(i)}_t$ tại từng iteration $i$ độc lập
- **Cấp 2** – *Per-instance LoRA Adaptation*: điều chỉnh nhẹ query encoder $E_q$ qua LoRA, reset sau mỗi test instance

---

## 2. Phân tích khoảng cách Training–Test

### 2.1. Cơ chế pseudo-GT của ReSCORE `[FACT — ReSCORE, §3.2, Eq. 1–2]`

Trong quá trình training, với mỗi QA pair $(q, a)$ và document $d^{(i)}_j$ (tài liệu $j$ tại iteration $i$), ReSCORE tính pseudo-GT label:

$$Q^{(i)}_{LM}\!\left(d^{(i)}_j \;\Big|\; q^{(i)}\right) \;\propto\; \underbrace{P^{(i)}_{LM}\!\left(q \;\Big|\; d^{(i)}_j\right)}_{\text{(A) Relevance}} \;\cdot\; \underbrace{P^{(i)}_{LM}\!\left(a \;\Big|\; q,\; d^{(i)}_j\right)}_{\text{(B) Consistency}} \tag{1}$$

**Giải thích từng thành phần:**

- **(A) Relevance** $P_{LM}(q \mid d^{(i)}_j)$: Xác suất LLM sinh ra **câu hỏi $q$** khi được cho document $d^{(i)}_j$. Nếu document không liên quan đến chủ đề câu hỏi, LLM khó có thể sinh lại đúng câu hỏi đó → xác suất thấp. Thành phần này lọc các tài liệu không cùng chủ đề.

- **(B) Consistency** $P_{LM}(a \mid q, d^{(i)}_j)$: Xác suất LLM sinh ra **câu trả lời đúng $a$** khi cho trước câu hỏi $q$ và document $d^{(i)}_j$. Nếu document chứa thông tin giải quyết câu hỏi, xác suất cao. Thành phần này đảm bảo tài liệu có nội dung trả lời được câu hỏi.

**Tại sao cần cả hai?** `[FACT — ReSCORE, §3.2]`: Chỉ dùng (B) alone dẫn đến **−23.8% recall** (Table 3 ReSCORE) vì false positives — tài liệu chứa token trùng với $a$ nhưng không thực sự liên quan (ví dụ: document "2006 FIFA World Cup" có xác suất cao với $a=\text{"2006"}$ dù không liên quan đến câu hỏi về Pixar). Chỉ dùng (A) alone cải thiện +5.37% recall nhưng thiếu tín hiệu về câu trả lời.

Sau khi normalize trên top-$M$ documents, $Q^{(i)}_{LM}$ được dùng làm **teacher distribution**. Retriever được train bằng KL divergence:

$$\mathcal{L}_{ReSCORE} = \sum_{n=1}^{N} \sum_{i=0}^{\eta_n} D_{KL}\!\left(Q^{(i)}_{LM}\!\left(D^{(i)} \;\Big|\; q^{(i)}_n\right) \;\Big\|\; P^{(i)}_R\!\left(D^{(i)} \;\Big|\; q^{(i)}_n\right)\right) \tag{2}$$

**Giải thích Eq. (2):**

- $Q^{(i)}_{LM}$ là **phân phối mục tiêu** (teacher): LLM nói "tài liệu nào quan trọng"
- $P^{(i)}_R$ là **phân phối của retriever** (student): retriever hiện tại đang đánh giá tài liệu nào cao
- $D_{KL}(Q \| P)$ đo khoảng cách giữa hai phân phối; minimize tức là kéo $P^{(i)}_R$ về gần $Q^{(i)}_{LM}$
- Tổng theo $n$ (toàn bộ training instances) và $i$ (toàn bộ iterations của mỗi instance)

**Vấn đề cốt lõi tại test time:** Eq. (1) đòi hỏi $a$ để tính $P_{LM}(a \mid q, d)$, nhưng tại test time $a$ **không có sẵn**. Đây là rào cản chính.

### 2.2. Bảng so sánh Training vs. Test

| Yếu tố | Training (ReSCORE) | Test time (TTA) |
|---|---|---|
| Ground-truth answer $a$ | **Có** | **Không có** |
| Query reformulation | LLM-driven (Thought-concat) | Như nhau |
| Tín hiệu pseudo-GT | $P_{LM}(q,a \mid d)$ | Cần thiết kế thay thế |
| Cập nhật tham số | Toàn bộ $E_q$ | Chỉ LoRA hoặc chỉ vector |
| Rủi ro forgetting | Không | Có (nếu dùng Level 2) |

---

## 3. Review chi tiết TOUR

> **Nguồn:** Sung et al. (2023), "Optimizing Test-Time Query Representations for Dense Retrieval", ACL 2023 Findings, §3.1–3.3.

### 3.1. Ý tưởng cốt lõi `[FACT — TOUR, §3.1]`

TOUR **không cập nhật tham số encoder** $E_q$ hay $E_c$. Thay vào đó, nó tối ưu hóa trực tiếp **giá trị của vector** $q_t \in \mathbb{R}^d$ — vector biểu diễn của câu truy vấn — theo từng test instance:

$$q_{t+1} \leftarrow q_t - \eta \;\frac{\partial \mathcal{L}(q_t,\; C^{q_t}_{1:k})}{\partial q_t} \tag{3}$$

Ở đây:
- $q_t$ là vector query sau $t$ bước cập nhật (khởi đầu $q_0 = E_q(q)$ — vector từ encoder)
- $\mathcal{L}$ là hàm loss được tính từ **top-$k$ retrieved documents tại bước $t$** (ký hiệu $C^{q_t}_{1:k}$, bởi vì mỗi bước $t$ có thể retrieve ra tập documents khác do $q_t$ đã thay đổi)
- $\eta$ là learning rate; gradient được tính theo $q_t$ (vector), không theo tham số mô hình

**Sự khác biệt then chốt:** Gradient descent thông thường cập nhật $\theta$ (trọng số mô hình). TOUR cập nhật $q_t$ (một vector $d$-chiều) — encoder bất biến.

### 3.2. Biến thể Hard: TOUR$_\text{hard}$ `[FACT — TOUR, §3.2, Eq. 8–10]`

**Bước 1: Xây dựng tập pseudo-positive $C^q_{hard}$**

Cross-encoder $\phi$ cho mỗi document trong top-$k$ một score. Score này được normalize thành xác suất qua softmax nhiệt độ $\tau$:

$$P_k(\tilde{c} = c^* \mid q, \phi) = \frac{\exp(\phi(q, \tilde{c})/\tau)}{\sum_{i=1}^{k} \exp(\phi(q, c_i)/\tau)} \tag{4}$$

> **Giải thích ký hiệu:** $P_k(\tilde{c} = c^* \mid q, \phi)$ đọc là "xác suất mà document $\tilde{c}$ thực sự là gold document $c^*$, theo đánh giá của cross-encoder $\phi$". Đây là **nhãn giả** — cross-encoder ước lượng $c^*$ nhưng có thể sai.

$C^q_{hard}$ là **tập nhỏ nhất** sao cho tổng xác suất đạt ngưỡng $p$ (tương tự nucleus sampling):

$$C^q_{hard} = \arg\min_{S \subseteq C^q_{1:k}} |S| \quad \text{s.t.} \quad \sum_{\tilde{c} \in S} P_k(\tilde{c} = c^* \mid q, \phi) \geq p \tag{4b}$$

Các document trong $C^q_{hard}$ là "pseudo-positive" — được đối xử như $c^*$ để tính loss.

**Bước 2: Loss function**

$$\mathcal{L}_{hard}(q,\; C^q_{1:k}) = -\log \sum_{\tilde{c} \in C^q_{hard}} P_k(\tilde{c} \mid q) \tag{5}$$

với **retriever distribution**:
$$P_k(\tilde{c} \mid q) = \frac{\exp(\text{sim}(q, \tilde{c}))}{\sum_{i=1}^{k} \exp(\text{sim}(q, c_i))} \tag{5b}$$

> **Giải thích Eq. (5):** Đây là **maximum marginal likelihood** — maximize tổng xác suất mà retriever đặt vào các pseudo-positive documents $\tilde{c} \in C^q_{hard}$. Tương tự DPR loss nhưng thay $c^*$ bằng $C^q_{hard}$.

> **Phân biệt $P_k(\tilde{c} \mid q)$ và $P_k(\tilde{c} = c^* \mid q, \phi)$:**
> - $P_k(\tilde{c} \mid q)$: phân phối của **retriever** (softmax trên similarity scores $\text{sim}(q, c)$)
> - $P_k(\tilde{c} = c^* \mid q, \phi)$: phân phối của **cross-encoder** (softmax trên cross-encoder scores $\phi(q, c)$)
> - Hai phân phối này **khác nhau** và có vai trò khác nhau trong loss

**Update rule và quan hệ với Rocchio** `[FACT — TOUR, Appendix A]`:

Áp dụng gradient descent (Eq. 3) với loss ở Eq. (5), ta có thể chứng minh update rule tương đương:

$$g(q_t, C^{q_t}_{1:k}) = q_t + \eta \sum_{\tilde{c} \in C^{q_t}_{hard}} P(\tilde{c}|q_t)\,(1 - P_k(\tilde{c}|q_t))\,\tilde{c} - \eta \sum_{\tilde{c} \in C^{q_t}_{hard}} \left[ P(\tilde{c}|q_t) \sum_{\substack{c \in C^{q_t}_{1:k} \\ c \neq \tilde{c}}} P_k(c|q_t)\,c \right] \tag{6}$$

> **Giải thích Eq. (6):** Query vector được kéo **về phía** pseudo-positive documents $\tilde{c}$ (số hạng thứ hai) và **ra xa** các documents không phải pseudo-positive (số hạng thứ ba). Đây chính là tinh thần của Rocchio algorithm cổ điển, nhưng weight của mỗi document được tính từ cross-encoder thay vì equal weighting.

> **Cụ thể, với $P(\tilde{c}|q_t)$**: đây là phân phối **chỉ trên tập** $C^{q_t}_{hard}$: $P(\tilde{c}|q_t) = \dfrac{\exp(\text{sim}(q_t, \tilde{c}))}{\sum_{\tilde{c}' \in C^{q_t}_{hard}} \exp(\text{sim}(q_t, \tilde{c}'))}$. Khác với $P_k$ là tính trên toàn bộ top-$k$.

### 3.3. Biến thể Soft: TOUR$_\text{soft}$ `[FACT — TOUR, §3.3, Eq. 12–13]`

Thay vì hard selection, TOUR$_\text{soft}$ dùng toàn bộ cross-encoder distribution làm soft target:

$$\mathcal{L}_{soft}(q_t,\; C^{q_t}_{1:k}) = D_{KL}\!\left(P(\cdot \mid q_t, \phi) \;\|\; P_k(\cdot \mid q_t)\right) = -\sum_{i=1}^{k} P(c_i \mid q_t, \phi) \log \frac{P_k(c_i \mid q_t)}{P(c_i \mid q_t, \phi)} \tag{7}$$

> **Giải thích Eq. (7):** Minimize KL divergence từ phân phối retriever $P_k$ đến phân phối cross-encoder $P(\cdot|q_t,\phi)$. Nói cách khác: kéo retriever để "nhìn" giống cross-encoder — document nào cross-encoder cho điểm cao, retriever cũng nên đặt xác suất cao.

> **Quan hệ với ReSCORE Eq. (2):** Cả hai đều dùng KL divergence với teacher distribution. Điểm khác: ReSCORE dùng $Q_{LM}$ (từ LLM, cần answer $a$) làm teacher, cập nhật tham số $\theta_{Eq}$. TOUR$_\text{soft}$ dùng $P(\cdot|q,\phi)$ (từ cross-encoder, không cần $a$) làm teacher, cập nhật vector $q_t$.

Update rule tương đương (TOUR, Appendix B):

$$g(q_t, C^{q_t}_{1:k}) = q_t + \eta \sum_{i=1}^{k} P(c_i \mid q_t, \phi)\,c_i - \eta \sum_{i=1}^{k} P_k(c_i \mid q_t)\,c_i \tag{8}$$

> **Giải thích Eq. (8):** Query vector dịch chuyển về phía centroid weighted bởi cross-encoder (số hạng thứ hai), và bị đẩy ra khỏi centroid weighted bởi retriever hiện tại (số hạng thứ ba). Nếu cross-encoder và retriever đồng thuận, hai số hạng triệt tiêu nhau → không cần cập nhật.

### 3.4. Hiệu quả thực nghiệm của TOUR `[FACT — TOUR, Table 1, Table 2]`

- Cải thiện EM lên đến **+10.7%** trên open-domain QA (in-domain)
- **Query distribution shift (OOD):** TOUR$_{hard, k=20}$ cải thiện **+6.5% EM trung bình** trên unseen distributions (Table 2, TOUR paper)
- Không cần ground truth tại test time → khả thi trong OOD setting
- Không cập nhật tham số encoder → zero catastrophic forgetting

---

## 4. Framework đề xuất: ReSCORE-TTA `[HYPOTHESIS]`

### 4.1. Tổng quan kiến trúc

```
Input: test query q (một test instance), không có answer a
             │
             ▼
   q^(1) = q (initial query)
             │
   ┌─────────▼──────────────────────────────────────────────────┐
   │  FOR each iteration i = 1, ..., η_q  (multi-hop loop)      │
   │                                                             │
   │  q^(i)_0 = E_q(q^(i);  θ_Eq + LoRA)   [frozen doc enc.]   │
   │             │                                               │
   │             ▼                                               │
   │    Top-M retrieval → D^(i) = {d_1,...,d_M}                 │
   │             │                                               │
   │      ┌──────┴───────┐                                       │
   │      ▼              ▼                                       │
   │  φ(q^(i), d_j)   P_LM(q^(i)|d_j)   ← [DUAL SIGNAL]        │
   │      └──────┬───────┘                                       │
   │             ▼                                               │
   │    Q̃_TTA ∝ φ · P_LM(q|d)   [pseudo-GT, no answer needed]   │
   │             │                                               │
   │      ┌──────┴───────┐                                       │
   │      ▼              ▼                                       │
   │  [Level 1]       [Level 2]                                  │
   │  Optimize q_t    Update LoRA                                │
   │  (vector only)   (parameters)                               │
   │      └──────┬───────┘                                       │
   │             ▼                                               │
   │    Top-k final retrieval với q^(i)_{T_inner}               │
   │             │                                               │
   │    LLM → a^(i) hoặc "unknown"                               │
   │    If a^(i) ≠ "unknown": RETURN a^(i)                      │
   │    Else: t^(i) = LLM_thought(); q^(i+1) = [t^(i); q^(i)]  │
   └─────────────────────────────────────────────────────────────┘
```

---

## 5. Thiết kế Pseudo-Label tại Test Time `[HYPOTHESIS trừ khi ghi rõ]`

Đây là **thách thức trung tâm**: không có $a$ tại test time để tính $P_{LM}(a \mid q, d)$.

### 5.1. Phân tích các lựa chọn tín hiệu

#### Option A: Cross-encoder only (theo TOUR) `[FACT — TOUR áp dụng, INFERENCE cho ReSCORE]`

$$\tilde{Q}^{(i)}_{CE}(d_j \mid q^{(i)}) = \frac{\exp(\phi(q^{(i)}, d_j)/\tau)}{\sum_{j'=1}^{M} \exp(\phi(q^{(i)}, d_{j'})/\tau)} \tag{9}$$

Đây chính là $P(c \mid q, \phi)$ trong TOUR, áp dụng cho iterative setting của ReSCORE.

- **Ưu điểm:** Không cần LLM call thêm; đã được kiểm chứng trong TOUR `[FACT]`
- **Nhược điểm** `[INFERENCE]`: Cross-encoder được train trên in-distribution data; trên OOD domain nó có thể kém đáng tin

#### Option B: LLM relevance-only (không cần $a$) `[INFERENCE từ ReSCORE Table 3]`

$$\tilde{Q}^{(i)}_{LM\text{-}rel}(d_j \mid q^{(i)}) \propto P_{LM}(q^{(i)} \mid d_j) \tag{10}$$

Đây chính là thành phần (A) trong Eq. (1) của ReSCORE, được dùng độc lập.

- **Bằng chứng từ ReSCORE `[FACT]`:** $P_{LM}(q \mid d)$ alone cải thiện recall **+5.37% trung bình** (Table 3, ReSCORE paper)
- **Ưu điểm:** LLM generalist → tốt hơn cross-encoder trên OOD domain `[INFERENCE]`
- **Nhược điểm `[FACT]`:** Thiếu consistency signal (thành phần B) → không lọc được false positives kiểu "token match không có relevance"

#### Option C: Self-predicted answer — **KHÔNG KHUYẾN NGHỊ** `[INFERENCE từ ReSCORE Table 3]`

Dùng LLM tự dự đoán $\hat{a}$ để thay cho $a$.

- `[FACT]`: $P_{LM}(a \mid q, d)$ alone **giảm −23.8% recall** (Table 3, ReSCORE paper) ngay cả với $a$ đúng
- `[INFERENCE]`: $\hat{a}$ là noisy answer → vấn đề còn nghiêm trọng hơn

#### Option D: Dual Pseudo-label — **Khuyến nghị** `[HYPOTHESIS]`

Kết hợp cross-encoder và LLM-relevance:

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) = \mathrm{Softmax}_j\!\left[\log \phi(q^{(i)}, d_j) + \log P_{LM}(q^{(i)} \mid d_j)\right] \tag{11}$$

Tường minh hơn:

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) = \frac{\phi(q^{(i)}, d_j) \cdot P_{LM}(q^{(i)} \mid d_j)}{\sum_{j'=1}^{M} \phi(q^{(i)}, d_{j'}) \cdot P_{LM}(q^{(i)} \mid d_{j'})} \tag{12}$$

**Cơ sở lý luận** `[INFERENCE]`:

| Thành phần | Điểm mạnh | Điểm yếu |
|---|---|---|
| $\phi(q, d)$ — Cross-encoder | Capture lexical + semantic match cục bộ, nhanh | OOD sensitivity; không capture multi-hop |
| $P_{LM}(q \mid d)$ — LLM relevance | LLM generalist, hiểu semantic sâu | Thiếu consistency; tốn compute |
| Kết hợp | Cross-encoder lọc irrelevant nhanh; LLM xác nhận semantic relevance | Tốn compute hơn Option A |

> **Lưu ý quan trọng:** Cả hai signal đều **không cần answer $a$** → khả thi tại test time.

### 5.2. Confidence-based Filtering `[HYPOTHESIS]`

Để tránh học từ pseudo-labels không đáng tin (đặc biệt trên OOD), chỉ sử dụng documents vượt ngưỡng confidence:

$$\text{Mask}^{(i)}_j = \mathbb{1}\!\left[\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \geq \theta_{conf}\right] \tag{13}$$

Documents có $\text{Mask}^{(i)}_j = 0$ bị loại khỏi loss computation.

---

## 6. Thiết kế hàm Loss chi tiết

### 6.1. Level 1: Query Vector Optimization `[HYPOTHESIS — mở rộng TOUR sang iterative MHQA]`

**Nguồn gốc ý tưởng:** Trực tiếp từ TOUR$_\text{soft}$ (Eq. 7), với hai điều chỉnh:
1. Thay $P(c_i \mid q_t, \phi)$ bằng $\tilde{Q}^{(i)}_{TTA}$ (dual pseudo-label thay vì CE-only)
2. Áp dụng độc lập cho từng iteration $i$ trong MHQA (vì $q^{(i)}$ thay đổi qua từng hop)

Với query $q^{(i)}$ tại iteration $i$, optimize vector $q^{(i)}_t$ qua $T_{inner}$ steps:

**Biến thể Hard** `[mở rộng TOUR$_\text{hard}$, Eq. 5]`:

$$C^{(i)}_{hard} = \left\{d_j \in D^{(i)} : d_j \text{ thuộc top-}p \text{ nucleus của } \tilde{Q}^{(i)}_{TTA}\right\} \tag{14}$$

$$\mathcal{L}^{(i)}_{\text{hard}} = -\log \sum_{d_j \in C^{(i)}_{hard}} P_k(d_j \mid q^{(i)}_t) \tag{15}$$

**Biến thể Soft — ưu tiên** `[mở rộng TOUR$_\text{soft}$, Eq. 7]`:

$$\mathcal{L}^{(i)}_{\text{soft}} = D_{KL}\!\left(\tilde{Q}^{(i)}_{TTA}(\cdot \mid q^{(i)}) \;\Big\|\; P^{(i)}_R(\cdot \mid q^{(i)}_t)\right) \tag{16}$$

$$= -\sum_{j=1}^{M} \tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \log \frac{P^{(i)}_R(d_j \mid q^{(i)}_t)}{\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)})} \tag{17}$$

> **Giải thích Eq. (16–17):** Cấu trúc **hoàn toàn tương đồng** với ReSCORE Eq. (2):
> - ReSCORE (training): $D_{KL}(Q^{(i)}_{LM} \| P^{(i)}_R)$ — teacher là LLM (cần $a$), student là retriever params
> - Level 1 (test time): $D_{KL}(\tilde{Q}^{(i)}_{TTA} \| P^{(i)}_R)$ — teacher là dual pseudo-label (không cần $a$), student là query vector $q^{(i)}_t$
>
> Sự thay thế $Q^{(i)}_{LM} \to \tilde{Q}^{(i)}_{TTA}$ là **điểm khác biệt cốt lõi** do không có $a$ tại test time.

**Update rule** `[TOUR Eq. 3]`:

$$q^{(i)}_{t+1} \leftarrow q^{(i)}_t - \eta_q \frac{\partial \mathcal{L}^{(i)}_{\text{soft}}}{\partial q^{(i)}_t} \tag{18}$$

Update rule tường minh (tương tự TOUR Appendix B, Eq. 8):

$$q^{(i)}_{t+1} = q^{(i)}_t + \eta_q \sum_{j=1}^{M} \tilde{Q}^{(i)}_{TTA}(d_j)\,d_j - \eta_q \sum_{j=1}^{M} P^{(i)}_R(d_j \mid q^{(i)}_t)\,d_j \tag{18b}$$

> **Giải thích Eq. (18b):** Query vector bị kéo về phía centroid của các documents được dual pseudo-label đánh giá cao (số hạng $+$), và bị đẩy ra khỏi centroid của các documents mà retriever hiện tại đang cho điểm cao (số hạng $-$). Nếu retriever đã align với pseudo-label, không cần cập nhật.

**Early stopping** `[FACT — TOUR §3.5]`: Dừng khi top-1 document của retriever thuộc $C^{(i)}_{hard}$. Từ TOUR: max $T_{inner} = 3$ iterations.

**Không có forgetting ở Level 1** `[FACT — TOUR, §3.1]`: Chỉ $q^{(i)}_t$ (vector) được cập nhật, không có tham số nào của encoder thay đổi.

### 6.2. Level 2: Per-instance LoRA Adaptation `[HYPOTHESIS]`

**Nguồn gốc ý tưởng:** LoRA (Hu et al., 2022) là kỹ thuật adapter tuning tiêu chuẩn. Áp dụng vào TTA là ý tưởng mới — không có trong TOUR hay ReSCORE.

**Kiến trúc LoRA** trên query encoder $E_q$:

$$W^{LoRA} = W_0 + \Delta W = W_0 + B \cdot A \tag{19}$$

- $W_0 \in \mathbb{R}^{d \times d}$: trọng số gốc của một linear layer trong $E_q$ (bất biến)
- $B \in \mathbb{R}^{d \times r}$: ma trận thứ nhất, khởi tạo $B = 0$ (để $\Delta W = 0$ lúc đầu)
- $A \in \mathbb{R}^{r \times d}$: ma trận thứ hai, khởi tạo $A \sim \mathcal{N}(0, \sigma^2)$
- $r \ll d$ (ví dụ $r=8$, $d=768$): rank thấp → ít tham số → bounded adaptation

Loss adaptation của LoRA `[HYPOTHESIS — tương tự ReSCORE Eq. 2 nhưng teacher khác]`:

$$\mathcal{L}^{(i)}_{\text{LoRA}} = D_{KL}\!\left(\tilde{Q}^{(i)}_{TTA}(\cdot \mid q^{(i)}) \;\Big\|\; P^{(i),\text{LoRA}}_R(\cdot \mid q^{(i)})\right) \tag{20}$$

> **So sánh với ReSCORE training loss Eq. (2):**
> - ReSCORE: cập nhật $\theta_{Eq}$ toàn bộ với teacher $Q^{(i)}_{LM}$ (cần $a$)
> - Level 2 TTA: cập nhật **chỉ** $\{B, A\}$ của LoRA với teacher $\tilde{Q}^{(i)}_{TTA}$ (không cần $a$)
>
> Cấu trúc loss giống nhau (KL divergence), nhưng (a) teacher signal khác nhau và (b) parameter được cập nhật hạn chế hơn nhiều.

**Per-instance reset protocol** `[HYPOTHESIS]`: Tại đầu mỗi test instance mới, reset $B = 0$. Đây là cơ chế chính ngăn catastrophic forgetting across instances.

### 6.3. Regularization chống Catastrophic Forgetting

#### Anchor Regularization `[INFERENCE — từ continual learning literature; e.g., EWC, Kirkpatrick et al. 2017]`

$$\mathcal{L}_{\text{anchor}} = \left\| q^{(i)}_t - q^{(i)}_0 \right\|_2^2 \tag{21}$$

> **Giải thích:** Kéo query vector không dịch chuyển quá xa khỏi điểm khởi đầu $q^{(i)}_0$. Tương tự "trust region" — cho phép cải thiện nhưng không cho phép deviation cực đoan. Được dùng trong continual learning để bảo toàn kiến thức cũ.

> **Tại sao cần điều này?** Nếu pseudo-labels $\tilde{Q}^{(i)}_{TTA}$ sai (noisy OOD), gradient descent không bị anchor có thể kéo $q^{(i)}_t$ đến vùng embedding space vô nghĩa.

#### LoRA Norm Regularization `[INFERENCE — standard LoRA practice]`

$$\mathcal{L}_{\text{LoRA-reg}} = \left\| B \cdot A \right\|_F^2 \tag{22}$$

> **Giải thích:** Giới hạn magnitude của $\Delta W = BA$. Frobenius norm nhỏ → LoRA không thay đổi $E_q$ quá nhiều so với $W_0$. Tương tự weight decay nhưng áp dụng riêng cho LoRA matrices.

#### Entropy Regularization (tùy chọn) `[HYPOTHESIS]`

$$\mathcal{L}_{\text{ent}} = -\sum_{j=1}^{M} P^{(i)}_R(d_j \mid q^{(i)}_t) \log P^{(i)}_R(d_j \mid q^{(i)}_t) \tag{23}$$

> **Giải thích:** Maximize entropy của retrieval distribution → tránh "collapse" về một document duy nhất (mode collapse). Đặc biệt quan trọng trong MHQA vì mỗi hop cần **nhiều documents khác nhau**.

### 6.4. Objective tổng hợp `[HYPOTHESIS]`

Tại test iteration $i$, objective đầy đủ:

$$\mathcal{L}^{(i)}_{\text{TTA}} = \underbrace{\mathcal{L}^{(i)}_{\text{soft}}}_{\substack{\text{Level 1} \\ \text{query vector} \\ \text{[TOUR-soft Eq.7]}}} + \underbrace{\alpha \cdot \mathcal{L}^{(i)}_{\text{LoRA}}}_{\substack{\text{Level 2} \\ \text{LoRA adapter} \\ \text{[Novel]}}} + \underbrace{\beta \cdot \mathcal{L}_{\text{anchor}}}_{\substack{\text{anti-forgetting} \\ \text{vector drift} \\ \text{[CL lit.]}}} + \underbrace{\gamma \cdot \mathcal{L}_{\text{LoRA-reg}}}_{\substack{\text{LoRA bound} \\ \text{param. drift} \\ \text{[LoRA practice]}}} \tag{24}$$

Tổng qua tất cả iterations của một test instance:

$$\mathcal{L}_{\text{TTA}}(q) = \sum_{i=1}^{\eta_q} \mathcal{L}^{(i)}_{\text{TTA}} \tag{25}$$

> **Giải thích từng thành phần trong Eq. (24):**
> - $\mathcal{L}^{(i)}_{\text{soft}}$: Tín hiệu học chính — kéo query vector về hướng dual pseudo-label (đối chiếu TOUR Eq. 7, ReSCORE Eq. 2)
> - $\alpha \cdot \mathcal{L}^{(i)}_{\text{LoRA}}$: Cùng tín hiệu học nhưng áp dụng vào LoRA parameters — cập nhật encoder nhẹ (đối chiếu ReSCORE Eq. 2 nhưng hạn chế ở LoRA)
> - $\beta \cdot \mathcal{L}_{\text{anchor}}$: Giữ query vector không lệch xa → robustness với noisy pseudo-labels
> - $\gamma \cdot \mathcal{L}_{\text{LoRA-reg}}$: Giữ LoRA không quá lớn → bounded parameter change

**Bảng nguồn gốc của từng loss thành phần:**

| Loss | Nguồn gốc ý tưởng | Trạng thái | Đối chiếu |
|---|---|---|---|
| $\mathcal{L}^{(i)}_{\text{soft}}$ | TOUR$_\text{soft}$ Eq. 7 | FACT (TOUR) + HYPOTHESIS (extension) | TOUR §3.3, ReSCORE §3.2 |
| $\mathcal{L}^{(i)}_{\text{LoRA}}$ | ReSCORE Eq. 2 + LoRA | HYPOTHESIS (novel combination) | ReSCORE §3.2 |
| $\mathcal{L}_{\text{anchor}}$ | EWC / trust region | INFERENCE từ CL literature | Kirkpatrick et al. 2017 |
| $\mathcal{L}_{\text{LoRA-reg}}$ | Standard LoRA training | INFERENCE từ LoRA practice | Hu et al. 2022 |
| Dual pseudo-label $\tilde{Q}^{(i)}_{TTA}$ | Tổng hợp TOUR + ReSCORE | HYPOTHESIS (novel) | TOUR Eq. 9, ReSCORE Table 3 |

---

## 7. Phân tích Catastrophic Forgetting

### 7.1. Tại sao đây là vấn đề trong TTA?

Trong TTA tuần tự (nhiều test instances liên tiếp), việc cập nhật tham số tích lũy có thể:
- Xóa khả năng trên domain cũ (*classic catastrophic forgetting*)
- Overfit vào noise của pseudo-labels OOD

### 7.2. Thiết kế ReSCORE-TTA giải quyết thế nào?

| Cơ chế | Loại bảo vệ | Tại đâu trong framework |
|---|---|---|
| Level 1 chỉ update vector $q_t$ — không update $\theta$ | **Zero forgetting** hoàn toàn | Query vector optimization |
| Per-instance LoRA reset ($B=0$) | Tránh **across-instance forgetting** | LoRA adaptation |
| Low-rank constraint (rank $r$) | Giới hạn không gian adaptation: $\text{dim}(\Delta W) \leq 2rd$ | LoRA architecture |
| Anchor regularization $\mathcal{L}_{\text{anchor}}$ | Tránh **within-instance over-adaptation** | Cả hai levels |
| LoRA norm bound $\mathcal{L}_{\text{LoRA-reg}}$ | Giới hạn magnitude thay đổi | LoRA weights |
| Confidence filtering $\theta_{conf}$ | Chỉ học từ pseudo-labels đáng tin | Pseudo-GT quality |

### 7.3. Phân tích lý thuyết `[INFERENCE]`

Gọi $\theta^*$ là tham số retriever sau training ReSCORE, $\theta^{LoRA}$ là tham số khi có LoRA:

$$\theta^{LoRA} = \theta^* + \Delta\theta^{LoRA} \quad\text{với}\quad \|\Delta\theta^{LoRA}\|_F \leq r \cdot \|B\|_F \cdot \|A\|_F \tag{26}$$

Vì per-instance reset đặt $B=0$ sau mỗi instance, **$\theta^*$ không bao giờ thực sự thay đổi persistent**. LoRA chỉ tồn tại trong phạm vi một test instance.

---

## 8. Phân tích chất lượng Pseudo-label

### 8.1. Nguồn gốc nhiễu

| Nguồn nhiễu | Từ thành phần nào | Mức độ nghiêm trọng |
|---|---|---|
| Cross-encoder OOD | $\phi(q, d)$ được train in-distribution | TRUNG BÌNH - CAO |
| Missing consistency signal | Thiếu $P_{LM}(a \mid q, d)$ vì không có $a$ | TRUNG BÌNH |
| LLM hallucination trong $P_{LM}(q \mid d)$ | LLM có thể score cao cho irrelevant docs | THẤP (LLM tốt hơn CE về OOD) |

**Bằng chứng `[FACT — ReSCORE Table 3]`:** Trên in-distribution data:
- $P_{LM}(q \mid d)$ alone: **+5.37% recall** trung bình — khá tốt khi không có $a$
- $P_{LM}(q, a \mid d)$: **+14.4% recall** — best, nhưng cần $a$

Kết luận `[INFERENCE]`: Dual pseudo-label Eq. (12) có thể đạt kết quả ở giữa hai con số này.

### 8.2. Chiến lược nâng cao chất lượng

**Strategy 1: Consensus check giữa hai tín hiệu** `[HYPOTHESIS]`

$$\text{Consensus}_j = \mathbb{1}\!\left[\phi(q, d_j) \geq \theta_{CE} \;\wedge\; P_{LM}(q \mid d_j) \geq \theta_{LM}\right] \tag{27}$$

Documents không có consensus bị loại hoặc down-weighted. Lý do: nếu một signal cho high score nhưng signal kia cho low score, document này đáng ngờ.

**Strategy 2: Thought-augmented pseudo-label** `[HYPOTHESIS — từ Thought-concat trong ReSCORE §4.2.4]`

Tại iteration $i \geq 2$, ta có thoughts $t^{(1)}, ..., t^{(i-1)}$. Dùng augmented query:

$$\tilde{q}^{(i)} = [t^{(i-1)};\; q^{(i)}] \tag{28}$$

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \propto \phi(\tilde{q}^{(i)}, d_j) \cdot P_{LM}(\tilde{q}^{(i)} \mid d_j) \tag{29}$$

> **Lý do `[INFERENCE]`:** Thought $t^{(i-1)}$ chứa partial answer từ hop trước — gián tiếp bổ sung consistency signal mà không cần $a$ trực tiếp. Đây là phiên bản "weak consistency" không cần ground truth.

**Strategy 3: Temperature calibration** `[HYPOTHESIS]`

$$\phi_{\text{cal}}(q, d) = \phi(q, d)\,/\,\tau_{\text{OOD}}, \quad \tau_{\text{OOD}} = -\sum_j \hat{p}_j \log \hat{p}_j \tag{30}$$

High entropy distribution → cross-encoder uncertain → tăng temperature để soften labels.

---

## 9. Kết nối với iterative structure của ReSCORE

### 9.1. Adaptation tại mỗi hop (Level 1) `[HYPOTHESIS]`

TOUR gốc được thiết kế cho single-hop. Trong MHQA, $q^{(i)}$ khác $q^{(i+1)}$ về ngữ nghĩa. Do đó, TOUR phải áp dụng **độc lập** tại mỗi hop — không có gradient flow giữa các hops:

$$q^{(i)}_{t+1} \leftarrow q^{(i)}_t - \eta_q \frac{\partial \mathcal{L}^{(i)}_{\text{soft}}(q^{(i)}_t,\; D^{(i)})}{\partial q^{(i)}_t} \tag{31}$$

Điều này khác với ReSCORE training (Eq. 2) nơi gradient flow qua nhiều iterations.

### 9.2. LoRA tích lũy within-instance (Level 2) `[HYPOTHESIS]`

LoRA **tích lũy qua các iterations** của một test instance, vì tất cả iterations phục vụ cùng câu hỏi $q$:

$$\Delta\theta_{LoRA} = \sum_{i=1}^{\eta_q} \nabla_{\theta_{LoRA}} \!\left(\mathcal{L}^{(i)}_{\text{LoRA}} + \beta\,\mathcal{L}_{\text{anchor}} + \gamma\,\mathcal{L}_{\text{LoRA-reg}}\right) \tag{32}$$

### 9.3. Bảo toàn tính chất MHR tăng dần `[INFERENCE]`

ReSCORE được thiết kế để $\text{MHR}_i$@k **tăng dần** qua iterations `[FACT — ReSCORE Table 2]`. TTA phải không phá vỡ tính chất này:

- Level 1 (per-iteration): Mỗi iteration $i$ dùng pseudo-label tốt hơn (thought-augmented) → $\text{MHR}_i$ không nên giảm `[INFERENCE]`
- Level 2 (LoRA tích lũy): Sau hop 1, LoRA đã học context của question → hop 2 search tốt hơn `[HYPOTHESIS]`

---

## 10. Thiết kế thực nghiệm

### 10.1. Research Questions

| RQ | Nội dung | Phương pháp kiểm tra |
|---|---|---|
| **RQ1** | Dual pseudo-label tốt hơn CE-only hay LM-rel-only? | Ablation: CE-only vs LM-rel-only vs Dual |
| **RQ2** | Level 1 alone vs. Level 2 alone vs. kết hợp? | Ablation: L1-only, L2-only, L1+L2 |
| **RQ3** | Hard vs. Soft variant trong MHQA? | Thay $\mathcal{L}_{soft}$ bằng $\mathcal{L}_{hard}$ |
| **RQ4** | TTA có làm hỏng MHR tăng dần không? | Plot $\text{MHR}_i$@k theo $i$ |
| **RQ5** | $\beta$ (anchor weight) ảnh hưởng thế nào? | Grid search $\beta \in \{0, 0.01, 0.1, 1.0\}$ |
| **RQ6** | LoRA rank $r$ tối ưu? | $r \in \{4, 8, 16, 32\}$ |

### 10.2. OOD Evaluation Setup

| Scenario | Train dataset | Test OOD | Lý do |
|---|---|---|---|
| Cross-dataset A | MuSiQue | HotpotQA, 2WikiMHQA | Khác loại reasoning |
| Cross-dataset B | HotpotQA | MuSiQue, 2WikiMHQA | |
| Cross-hop | 2-hop (MuSiQue subset) | 3-hop (MuSiQue subset) | Khác độ phức tạp |

### 10.3. Baselines

| Model | Mô tả | Mục đích |
|---|---|---|
| ReSCORE (no TTA) | Baseline chính | Điểm so sánh gốc |
| ReSCORE + TOUR$_{hard}$ (CE-only) | Level 1, hard, $\tilde{Q}=\phi$ | Ablation: CE-only |
| ReSCORE + TOUR$_{soft}$ (CE-only) | Level 1, soft, $\tilde{Q}=\phi$ | Ablation: CE-only |
| ReSCORE + L1 (LM-rel-only) | Level 1, soft, $\tilde{Q}=P_{LM}(q\|d)$ | Ablation: LM-only |
| **ReSCORE-TTA (L1)** | Level 1, Dual pseudo-label | Ours |
| **ReSCORE-TTA (L1+L2)** | Level 1 + Level 2 (LoRA) | Ours (full) |

### 10.4. Metrics

- **QA:** EM, F1 `[FACT — ReSCORE §4.1]`
- **Retrieval:** $\text{MHR}_i$@$k$ cho $i \in \{1, 2, \eta_n\}$, $k=8$ `[FACT — ReSCORE §4.1, Eq. 3]`
- **Efficiency:** Seconds/query so với ReSCORE no-TTA

### 10.5. Hyperparameters

| Hyperparameter | Range | Giá trị suggest | Nguồn |
|---|---|---|---|
| $T_{inner}$ | 1–3 | 3 | `[FACT — TOUR §3.5]` |
| $\eta_q$ (query LR) | 0.5–2.0 | 1.2 | `[FACT — TOUR Table 7]` |
| $\eta_{LoRA}$ | 1e-4 – 1e-3 | 5e-4 | `[INFERENCE — standard LoRA]` |
| $r$ (LoRA rank) | 4, 8, 16 | 8 | `[HYPOTHESIS]` |
| $\tau$ (CE temp.) | 0.1–1.0 | 0.5 | `[FACT — TOUR Table 7]` |
| $p$ (nucleus) | 0.3–0.7 | 0.5 | `[FACT — TOUR Table 7]` |
| $\beta$ (anchor) | 0.01–1.0 | 0.1 | `[HYPOTHESIS]` |
| $\gamma$ (LoRA-reg) | 0.01–0.1 | 0.01 | `[HYPOTHESIS]` |
| $M$ (candidate pool) | 32 | 32 | `[FACT — ReSCORE §4.1]` |
| $k$ (inference top-k) | 8 | 8 | `[FACT — ReSCORE §4.1]` |

---

## 11. Phân tích rủi ro

| Rủi ro | Mức độ | Mitigation | Fallback |
|---|---|---|---|
| Pseudo-label noise OOD | **CAO** | Confidence filtering (Eq. 13) + consensus (Eq. 27) | Chỉ dùng Level 1 + CE-only |
| LoRA overfit trong-instance | TRUNG BÌNH | $\mathcal{L}_{anchor}$ + $\mathcal{L}_{LoRA-reg}$ | Disable Level 2, chỉ Level 1 |
| Latency quá cao | TRUNG BÌNH | Cache $\phi$; early stopping | Chỉ Level 1 |
| TOUR không phù hợp iterative | TRUNG BÌNH | Per-iteration independence (Eq. 31) | Áp dụng chỉ tại hop 1 |
| $P_{LM}(q\|d)$ compute cost (32 LLM calls/iter) | **CAO** | Batch processing; smaller model | Drop LM signal, chỉ CE |

---

## 12. Tóm tắt framework

```
ReSCORE-TTA = ReSCORE (pretrained) 
            + TOUR-soft với Dual Pseudo-label (Level 1)
            + Per-instance LoRA Adaptation (Level 2)

Pseudo-GT tại test time (không cần answer a):
    Q̃_TTA ∝ φ(q,d) · P_LM(q|d)         [Eq. 12]

Level 1 — zero forgetting:
    Optimize query vector q_t via gradient descent
    Loss: L_soft = KL(Q̃_TTA || P_R(·|q_t))  [Eq. 16 ~ TOUR Eq. 7]

Level 2 — bounded forgetting (per-instance reset):
    Optimize LoRA(E_q) via gradient descent
    Loss: L_LoRA = KL(Q̃_TTA || P_R^LoRA)    [Eq. 20 ~ ReSCORE Eq. 2]

Combined objective:
    L_TTA = L_soft + α·L_LoRA              [Eq. 24]
          + β·L_anchor + γ·L_LoRA-reg      [anti-forgetting]
```

---

## Phân loại nguồn gốc ý tưởng (tóm tắt)

| Thành phần | Phân loại | Nguồn |
|---|---|---|
| Iterative RAG framework | FACT | ReSCORE §3.1 |
| KL divergence training loss | FACT | ReSCORE §3.2, Eq. 2 |
| Pseudo-GT từ $P_{LM}(q,a\|d)$ | FACT | ReSCORE §3.2, Eq. 1 |
| Query vector optimization | FACT | TOUR §3.1, Eq. 3 |
| TOUR$_\text{hard}$ loss | FACT | TOUR §3.2, Eq. 9 |
| TOUR$_\text{soft}$ loss (KL) | FACT | TOUR §3.3, Eq. 12 |
| Cross-encoder pseudo-labels | FACT | TOUR §3.1 |
| $P_{LM}(q\|d)$ signal quality (+5.37%) | FACT | ReSCORE Table 3 |
| Áp dụng TOUR cho iterative setting | INFERENCE | Mở rộng TOUR hợp lý |
| Anchor regularization | INFERENCE | CL literature |
| Dual pseudo-label $\tilde{Q}_{TTA}$ | **HYPOTHESIS** | Kết hợp TOUR + ReSCORE |
| Per-instance LoRA Adaptation | **HYPOTHESIS** | Mới hoàn toàn |
| Thought-augmented pseudo-label | **HYPOTHESIS** | Từ Thought-concat ReSCORE |

---

*Tài liệu này dựa hoàn toàn trên nội dung của ReSCORE (ACL 2025) và TOUR (ACL 2023 Findings). Mọi số liệu và kết quả thực nghiệm được trích dẫn từ hai paper này. Các ý tưởng được phân loại rõ ràng theo research protocol: `[FACT]` / `[INFERENCE]` / `[HYPOTHESIS]`.*
