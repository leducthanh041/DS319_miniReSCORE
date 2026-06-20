# Test-Time Adaptation for ReSCORE-based Iterative Retrieval

**Loại tài liệu:** Research Design Document  
**Paper nền tảng:** ReSCORE (ACL 2025), TOUR (ACL 2023 Findings)  
**Mục tiêu:** Cải thiện OOD generalization của ReSCORE mà không retraining toàn bộ retriever

---

## 1. Tóm tắt điều hành

ReSCORE (ACL 2025) huấn luyện một dense retriever cho iterative MHQA bằng pseudo-GT labels từ LLM, đạt SOTA trên MuSiQue / HotpotQA / 2WikiMHQA. Tuy nhiên, các tác giả **tự thừa nhận hạn chế OOD**: *"its ability to generalize to other datasets that differ in reasoning patterns or dataset characteristics remains limited."*

TOUR (ACL 2023 Findings) đề xuất tối ưu hóa **vector biểu diễn câu truy vấn** tại test time thông qua gradient descent có hướng dẫn từ cross-encoder reranker, hoàn toàn không cập nhật tham số encoder.

Tài liệu này đề xuất **ReSCORE-TTA**: một framework hai cấp độ kết hợp:

- **Cấp 1** – *Query Vector Optimization* (theo tinh thần TOUR): tối ưu hóa vector $q^{(i)}$ tại từng iteration
- **Cấp 2** – *Per-instance LoRA Adaptation*: điều chỉnh nhẹ query encoder qua LoRA, reset sau mỗi test query

---

## 2. Phân tích khoảng cách Training–Test

### 2.1. Cơ chế pseudo-GT của ReSCORE

Trong quá trình training, ReSCORE tính pseudo-GT label như sau:

$$Q^{(i)}_{LM}(d^{(i)}_j \mid q^{(i)}) \;\propto\; P^{(i)}_{LM}(q \mid d^{(i)}_j) \;\cdot\; P^{(i)}_{LM}(a \mid q,\, d^{(i)}_j) \tag{1}$$

Thành phần thứ nhất ($P_{LM}(q \mid d)$) đo **relevance** (tài liệu có liên quan đến câu hỏi không?).  
Thành phần thứ hai ($P_{LM}(a \mid q, d)$) đo **consistency** (tài liệu có giúp trả lời đúng không?).

Retriever được train bằng KL divergence:

$$\mathcal{L}_{ReSCORE} = \sum_{n=1}^{N} \sum_{i=0}^{\eta_n} D_{KL}\!\left(Q^{(i)}_{LM}(D^{(i)} \mid q^{(i)}_n) \;\|\; P^{(i)}_R(D^{(i)} \mid q^{(i)}_n)\right) \tag{2}$$

**Vấn đề cốt lõi tại test time:** Công thức (1) đòi hỏi câu trả lời đúng $a$, nhưng tại test time $a$ **không có sẵn**. Đây là rào cản chính để áp dụng trực tiếp tín hiệu pseudo-GT của ReSCORE vào TTA.

### 2.2. Bảng so sánh Training vs. Test

| Yếu tố | Training (ReSCORE) | Test time (TTA) |
|---|---|---|
| Ground-truth answer $a$ | **Có** | **Không có** |
| Query reformulation | LLM-driven (Thought-concat) | Như nhau |
| Pseudo-GT tín hiệu | $P_{LM}(q,a \mid d)$ | Cần thiết kế thay thế |
| Cập nhật tham số | Toàn bộ query encoder | Chỉ LoRA (hoặc chỉ vector) |
| Rủi ro forgetting | Không | Có (nếu dùng Level 2) |

---

## 3. Review chi tiết TOUR (Nguồn: Paper thực tế)

### 3.1. Ý tưởng cốt lõi

TOUR (Test-Time Optimization of Query Representations) **không cập nhật tham số encoder**. Thay vào đó, nó tối ưu hóa trực tiếp vector $q_t \in \mathbb{R}^d$ theo từng test instance:

$$q_{t+1} \leftarrow q_t - \eta \frac{\partial \mathcal{L}(q_t, C^{q_t}_{1:k})}{\partial q_t} \tag{3}$$

Pseudo-label được cung cấp bởi cross-encoder reranker $\phi(q, c)$, không cần ground truth.

### 3.2. TOUR$_{\text{hard}}$

Xây dựng tập pseudo-positive $C^q_{hard}$ là tập nhỏ nhất thỏa mãn:

$$\sum_{\tilde{c} \in C^q_{hard}} P_k(\tilde{c} = c^* \mid q, \phi) \geq p, \quad \text{với } P_k(\tilde{c}=c^* \mid q,\phi) = \frac{\exp(\phi(q,\tilde{c})/\tau)}{\sum_{i=1}^k \exp(\phi(q,c_i)/\tau)} \tag{4}$$

Hàm loss:

$$\mathcal{L}_{hard}(q, C^q_{1:k}) = -\log \sum_{\tilde{c} \in C^q_{hard}} P_k(\tilde{c} \mid q) \tag{5}$$

với $P_k(\tilde{c} \mid q) = \frac{\exp(\text{sim}(q, \tilde{c}))}{\sum_{i=1}^k \exp(\text{sim}(q, c_i))}$.

Update rule tương đương Rocchio algorithm tổng quát (được chứng minh trong Appendix A của TOUR):

$$g(q_t, C^{q_t}_{1:k}) = q_t + \eta \sum_{\tilde{c}} P(\tilde{c}|q_t)(1-P_k(\tilde{c}|q_t))\tilde{c} - \eta \sum_{\tilde{c}} \left[ P(\tilde{c}|q_t) \sum_{c \neq \tilde{c}} P_k(c|q_t)c \right] \tag{6}$$

### 3.3. TOUR$_{\text{soft}}$

Thay hard selection bằng soft distribution từ cross-encoder. Loss là KL divergence:

$$\mathcal{L}_{soft}(q_t, C^{q_t}_{1:k}) = -\sum_{i=1}^k P(c_i \mid q_t, \phi) \log \frac{P_k(c_i \mid q_t)}{P(c_i \mid q_t, \phi)} \tag{7}$$

Update rule tương đương:

$$g(q_t, C^{q_t}_{1:k}) = q_t + \eta \sum_{i=1}^k P(c_i \mid q_t, \phi)\, c_i - \eta \sum_{i=1}^k P_k(c_i \mid q_t)\, c_i \tag{8}$$

Trực giác: query vector bị kéo về phía centroid weighted bởi cross-encoder, và bị đẩy ra khỏi centroid weighted bởi retriever hiện tại.

### 3.4. Hiệu quả thực nghiệm của TOUR

- Cải thiện EM lên đến **+10.7%** trên open-domain QA
- Hoạt động tốt đặc biệt khi **query distribution shift** (OOD): TOUR$_{hard, k=20}$ cải thiện **+6.5% EM trung bình** trên unseen distributions
- **Không cần ground truth** tại test time
- **Không cập nhật tham số** → zero catastrophic forgetting

---

## 4. Framework đề xuất: ReSCORE-TTA

### 4.1. Tổng quan kiến trúc

```
Input: test query q, iteration i
           │
           ▼
[Frozen Document Encoder]   [Query Encoder (LoRA)]
           │                          │
           ▼                          ▼
    Document Vectors          q^(i)_0 = Embed_query(q^(i))
           │                          │
           └──────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Top-M Retrieval     │
            │  (ANN search)        │
            └──────────┬───────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  ┌───────────────┐       ┌──────────────────┐
  │  Cross-encoder│       │  LLM Relevance   │
  │  Reranker φ   │       │  P_LM(q^(i)|d_j) │
  └───────┬───────┘       └────────┬─────────┘
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Dual Pseudo-label      │
          │  Q̃_TTA = φ * P_LM(q|d) │
          └─────────┬───────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
  ┌───────────────┐   ┌────────────────────┐
  │ Level 1:      │   │ Level 2:           │
  │ Query Vector  │   │ LoRA Adapter       │
  │ Optimization  │   │ Adaptation         │
  │ (TOUR-style)  │   │ (per-instance)     │
  └───────┬───────┘   └─────────┬──────────┘
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Anti-forgetting    │
          │  Regularization     │
          └──────────┬──────────┘
                     │
                     ▼
           Updated q^(i)_{t+1}
```

---

## 5. Thiết kế Pseudo-Label tại Test Time

Đây là **thách thức trung tâm** của TTA cho ReSCORE: không có $a$ tại test time để tính $P_{LM}(a \mid q, d)$.

### 5.1. Phân tích các lựa chọn tín hiệu

#### Option A: Cross-encoder only (TOUR approach)

$$\tilde{Q}^{(i)}_{CE}(d_j \mid q^{(i)}) = \frac{\exp(\phi(q^{(i)}, d_j)/\tau)}{\sum_{j'} \exp(\phi(q^{(i)}, d_{j'})/\tau)} \tag{9}$$

- **Ưu điểm:** Không cần LLM, nhanh  
- **Nhược điểm:** Cross-encoder cũng có thể yếu trên OOD; không capture multi-hop reasoning

#### Option B: Relevance-only LLM (không cần $a$)

$$\tilde{Q}^{(i)}_{LM-rel}(d_j \mid q^{(i)}) \propto P_{LM}(q^{(i)} \mid d_j) \tag{10}$$

- **Ưu điểm:** LLM đã có trong hệ thống, capture semantic relevance  
- **Bằng chứng từ ReSCORE (Table 3):** $P_{LM}(q \mid d)$ alone cải thiện recall **+5.37% trung bình** so với baseline  
- **Nhược điểm:** Thiếu consistency signal → có thể tăng false positives ít hơn (nhưng không triệt để)

#### Option C: Self-predicted answer (rủi ro cao)

Dùng LLM tự dự đoán $\hat{a}$ làm noisy answer. **KHÔNG KHUYẾN NGHỊ** vì:

- Nếu LLM sai, pseudo-GT sẽ nhiễu nghiêm trọng (error propagation)
- ReSCORE Table 3 chứng minh $P_{LM}(a \mid q, d)$ alone **giảm 23.8%** do false positives → noisy answer sẽ càng tệ hơn

#### Option D (**Khuyến nghị**): Dual Pseudo-label

Kết hợp cross-encoder và LLM-relevance:

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \propto \phi(q^{(i)}, d_j) \cdot P_{LM}(q^{(i)} \mid d_j) \tag{11}$$

Sau khi normalize (Softmax):

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) = \frac{\exp\!\left(\log \phi(q^{(i)}, d_j) + \log P_{LM}(q^{(i)} \mid d_j)\right)}{\sum_{j'} \exp\!\left(\log \phi(q^{(i)}, d_{j'}) + \log P_{LM}(q^{(i)} \mid d_{j'})\right)} \tag{12}$$

**Lý do kết hợp:**

| Thành phần | Vấn đề riêng lẻ | Lợi ích khi kết hợp |
|---|---|---|
| $\phi(q, d)$ | OOD sensitivity, chỉ local lexical/semantic | Cung cấp ranking signal nhanh |
| $P_{LM}(q \mid d)$ | Thiếu consistency | Lọc topically irrelevant docs |
| Kết hợp | — | Cross-encoder kiểm tra relevance cục bộ; LLM kiểm tra relevance sâu hơn |

**Lưu ý:** Cả hai signal đều **không cần answer $a$** → khả thi tại test time.

### 5.2. Confidence-based Pseudo-label Filtering

Để tránh pseudo-label nhiễu, chỉ sử dụng pseudo-label từ các documents có độ tin cậy cao:

$$\text{Mask}^{(i)}_j = \mathbb{1}\left[\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \geq \theta_{conf}\right] \tag{13}$$

Documents không vượt ngưỡng $\theta_{conf}$ bị loại khỏi loss computation. Điều này đặc biệt quan trọng trên OOD data khi cross-encoder kém tin cậy.

---

## 6. Hàm Loss chi tiết

### 6.1. Level 1: Query Vector Optimization

Cho mỗi iteration $i$ tại test time, optimize query vector $q^{(i)}_t$ (bắt đầu từ $q^{(i)}_0 = Embed_{query}(q^{(i)})$):

**Biến thể Hard (TOUR$_{hard}$ cho MHQA):**

$$C^{(i)}_{hard} = \left\{ d_j \in D^{(i)} : \text{top-}p \text{ documents under } \tilde{Q}^{(i)}_{TTA} \right\} \tag{14}$$

$$\mathcal{L}^{(i)}_{\text{hard}} = -\log \sum_{d_j \in C^{(i)}_{hard}} P_k(d_j \mid q^{(i)}_t) \tag{15}$$

**Biến thể Soft (ưu tiên cho iterative setting):**

$$\mathcal{L}^{(i)}_{\text{soft}} = D_{KL}\!\left(\tilde{Q}^{(i)}_{TTA}(\cdot \mid q^{(i)}) \;\|\; P^{(i)}_R(\cdot \mid q^{(i)}_t)\right) \tag{16}$$

$$= -\sum_{j=1}^{M} \tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \log \frac{P^{(i)}_R(d_j \mid q^{(i)}_t)}{\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)})} \tag{17}$$

**Update rule (gradient descent trên vector):**

$$q^{(i)}_{t+1} \leftarrow q^{(i)}_t - \eta_q \frac{\partial \mathcal{L}^{(i)}_{\text{soft}}}{\partial q^{(i)}_t} \tag{18}$$

**Lưu ý quan trọng:** Chỉ $q^{(i)}_t$ được cập nhật, encoder parameters $\theta_{Eq}$ không thay đổi → **zero catastrophic forgetting** ở Level 1.

**Early stopping** (theo TOUR): dừng khi top-1 retrieved document thuộc $C^{(i)}_{hard}$, hoặc khi top-1 có score cao nhất từ cross-encoder.

### 6.2. Level 2: Per-instance LoRA Adaptation

LoRA được thêm vào **chỉ các attention layers** của query encoder (các layers cao nhất, ví dụ 2–4 layers cuối):

$$W^{LoRA} = W_0 + \Delta W = W_0 + B \cdot A \tag{19}$$

trong đó $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, rank $r \ll d$ (ví dụ $r = 8$).

Loss adaptation của LoRA:

$$\mathcal{L}^{(i)}_{\text{LoRA}} = D_{KL}\!\left(\tilde{Q}^{(i)}_{TTA}(\cdot \mid q^{(i)}) \;\|\; P^{(i), \text{LoRA}}_R(\cdot \mid q^{(i)})\right) \tag{20}$$

**Per-instance reset protocol:** Tại đầu mỗi test instance, $B$ và $A$ được reset về $B = 0$, $A \sim \mathcal{N}(0, \sigma^2)$ (theo khởi tạo LoRA chuẩn). Điều này đảm bảo:
- Không tích lũy gradient giữa các test instances khác nhau
- Không có catastrophic forgetting across-instance

### 6.3. Regularization chống Catastrophic Forgetting

Mặc dù per-instance reset ngăn forgetting xuyên instance, cần thêm regularization để tránh **within-instance over-adaptation** (overfit vào pseudo-label nhiễu của một query đơn lẻ):

**Anchor regularization (đơn giản, hiệu quả):**

$$\mathcal{L}_{\text{anchor}} = \lambda_1 \left\| q^{(i)}_t - q^{(i)}_0 \right\|_2^2 \tag{21}$$

Kéo query vector không lệch quá xa khỏi điểm ban đầu.

**LoRA norm regularization:**

$$\mathcal{L}_{\text{LoRA-reg}} = \lambda_2 \left\| B \cdot A \right\|_F^2 \tag{22}$$

Giới hạn độ lớn của adaptation, tránh parameter shift cực đoan.

**Entropy regularization** (tùy chọn, tránh collapse):

$$\mathcal{L}_{\text{ent}} = -\lambda_3 \sum_{j=1}^M P^{(i)}_R(d_j \mid q^{(i)}_t) \log P^{(i)}_R(d_j \mid q^{(i)}_t) \tag{23}$$

Khuyến khích retrieval distribution không bị collapse về một document duy nhất.

### 6.4. Objective tổng hợp

Tại test iteration $i$, objective đầy đủ:

$$\mathcal{L}^{(i)}_{\text{TTA}} = \underbrace{\mathcal{L}^{(i)}_{\text{soft}}}_{\text{Level 1: query vector}} + \underbrace{\alpha \cdot \mathcal{L}^{(i)}_{\text{LoRA}}}_{\text{Level 2: LoRA}} + \underbrace{\beta \cdot \mathcal{L}_{\text{anchor}}}_{\text{anti-forgetting}} + \underbrace{\gamma \cdot \mathcal{L}_{\text{LoRA-reg}}}_{\text{LoRA bound}} \tag{24}$$

Toàn bộ $T$ iterations cho một query $q$:

$$\mathcal{L}_{\text{TTA}}(q) = \sum_{i=1}^{\eta_q} \mathcal{L}^{(i)}_{\text{TTA}} \tag{25}$$

với $\eta_q$ là số iterations được LLM xác định (tương tự ReSCORE training).

**Quy trình tối ưu hóa:**

```
For each test instance (q, unk_answer):
    Reset: LoRA weights → (B=0, A~Normal)
    q^(1) = q  (initial query)
    
    For i = 1, ..., η_q:
        q^(i)_0 = Embed_query(q^(i); θ_Eq + LoRA)
        
        # Retrieve top-M documents
        D^(i) = top-M retrieval with q^(i)_0
        
        # Compute dual pseudo-labels (no answer needed)
        CE_scores = φ(q^(i), d_j) for d_j in D^(i)
        LM_rel = P_LM(q^(i) | d_j) for d_j in D^(i)
        Q̃_TTA = Softmax(log(CE) + log(LM_rel))
        
        # Level 1: Optimize query vector (t steps)
        for t = 1, ..., T_inner:
            q^(i)_t = q^(i)_{t-1} - η_q * ∂L_soft/∂q^(i)_{t-1}
            [check early stopping]
        
        # Level 2: Update LoRA (1 gradient step)
        LoRA ← LoRA - η_LoRA * ∇(L_LoRA + β·L_anchor + γ·L_LoRA-reg)
        
        # Retrieve top-k with updated query
        D^(i)_final = top-k retrieval with q^(i)_T
        
        # LLM answer prediction & query reformulation
        a^(i) = LLM(D^(i)_final, thoughts)
        if a^(i) ≠ "unknown": return a^(i)
        t^(i) = LLM_thought(D^(i)_final, q^(i))
        q^(i+1) = [t^(i); q^(i)]  # Thought-concat
```

---

## 7. Phân tích Catastrophic Forgetting

### 7.1. Tại sao đây là vấn đề nghiêm trọng trong TTA?

Trong standard TTA, mô hình thường được adapt trên một luồng data liên tục. Nếu các test queries từ các domains khác nhau đến lần lượt, việc update tham số tích lũy có thể:

- **Xóa khả năng trên domain cũ** (classic catastrophic forgetting)
- **Overfit vào noise** của pseudo-labels OOD

### 7.2. Thiết kế ReSCORE-TTA ngăn forgetting thế nào?

| Cơ chế | Loại bảo vệ | Áp dụng ở đâu |
|---|---|---|
| **Level 1 chỉ update vector** | Hoàn toàn tránh forgetting | Query vector optimization |
| **Per-instance LoRA reset** | Tránh across-instance forgetting | LoRA adaptation |
| **Low-rank constraint** (rank $r$) | Giới hạn không gian adaptation | LoRA kiến trúc |
| **Anchor regularization** $\mathcal{L}_{anchor}$ | Tránh within-instance over-adaptation | Cả hai levels |
| **LoRA norm bound** $\mathcal{L}_{LoRA-reg}$ | Giới hạn magnitude thay đổi | LoRA weights |
| **Confidence filtering** $\theta_{conf}$ | Chỉ học từ pseudo-labels đáng tin | Pseudo-GT quality |

### 7.3. Phân tích lý thuyết

**Claim (INFERENCE):** Per-instance LoRA reset + anchor regularization đảm bảo bounded parameter deviation.

Gọi $\theta^*$ là tham số retriever sau training (ReSCORE), $\theta^{(q)}$ là tham số sau TTA cho query $q$:

$$\|\theta^{(q)} - \theta^*\|_2 \leq \underbrace{r \cdot \|B\|_F \cdot \|A\|_F}_{\text{bounded by LoRA-reg}} \tag{26}$$

Vì LoRA reset sau mỗi instance, $\theta^*$ không bao giờ thực sự thay đổi persistent. LoRA chỉ tồn tại trong phạm vi một test instance.

**Rủi ro còn lại:** Within-instance, nếu $\eta_q$ lớn (nhiều hops) và pseudo-labels nhiễu cao, LoRA có thể overfit. Anchor regularization kiểm soát điều này.

---

## 8. Phân tích chất lượng Pseudo-label

### 8.1. Nguồn gốc nhiễu pseudo-label

Tại test time, tín hiệu chúng ta có là:

1. **Cross-encoder $\phi(q, d)$**: Có thể kém chính xác trên OOD vì cross-encoder cũng được train trên domain cụ thể.
2. **$P_{LM}(q \mid d)$**: Theo ReSCORE Table 3, signal này cải thiện recall +5.37% – tương đối đáng tin, nhưng không kết hợp consistency.

**FACT (từ ReSCORE, Table 3):** Trên in-distribution data:
- $P_{LM}(q \mid d)$ alone: **+5.37% recall** trung bình
- $P_{LM}(a \mid q, d)$ alone: **-23.8% recall** (vì false positives)
- $P_{LM}(q, a \mid d)$ combined: **+14.4% recall** (best)

Kết luận: Khi không có $a$, $P_{LM}(q \mid d)$ là tín hiệu tốt nhất available. Việc bổ sung $\phi(q, d)$ là hypothesis chưa được kiểm chứng nhưng có cơ sở hợp lý.

### 8.2. Chiến lược nâng cao chất lượng pseudo-label

**Strategy 1: Consistency check giữa hai tín hiệu**

Chỉ giữ documents mà cả cross-encoder VÀ LLM-relevance đều đồng ý là relevant:

$$\text{Consensus}_j = \mathbb{1}\left[\phi(q, d_j) \geq \theta_{CE} \;\wedge\; P_{LM}(q \mid d_j) \geq \theta_{LM}\right] \tag{27}$$

Documents không có consensus bị loại hoặc down-weighted.

**Strategy 2: Iterative signal aggregation**

Tại iteration $i \geq 2$, ta có thêm context từ thoughts $t^{(1)}, ..., t^{(i-1)}$. Dùng augmented query cho pseudo-label:

$$\tilde{q}^{(i)} = [t^{(i-1)}; q^{(i)}] \tag{28}$$

$$\tilde{Q}^{(i)}_{TTA}(d_j \mid q^{(i)}) \propto \phi(\tilde{q}^{(i)}, d_j) \cdot P_{LM}(\tilde{q}^{(i)} \mid d_j) \tag{29}$$

Lý do: Thoughts cung cấp partial answer context, gián tiếp bổ sung consistency signal mà không cần $a$ trực tiếp.

**Strategy 3: Temperature calibration**

Trên OOD data, cross-encoder thường produce over-confident hoặc under-confident scores. Hiệu chỉnh:

$$\phi_{\text{cal}}(q, d) = \phi(q, d) / \tau_{OOD} \tag{30}$$

$\tau_{OOD}$ có thể được estimate từ entropy của distribution: $\tau_{OOD} = -\sum_j \hat{p}_j \log \hat{p}_j$ (high entropy → high $\tau$).

---

## 9. Kết nối với iterative structure của ReSCORE

Điểm đặc biệt quan trọng: ReSCORE hoạt động trong **iterative setting**, không phải single-hop. TOUR được thiết kế cho single-hop retrieval. Cần điều chỉnh:

### 9.1. Adaptation at each hop

Tại mỗi iteration $i$, query $q^{(i)}$ là **khác nhau** (do Thought-concat reformulation). TOUR phải được áp dụng độc lập tại mỗi hop:

$$q^{(i)}_{t+1} \leftarrow q^{(i)}_t - \eta_q \frac{\partial \mathcal{L}^{(i)}_{\text{soft}}(q^{(i)}_t, D^{(i)})}{\partial q^{(i)}_t} \tag{31}$$

Không có gradient flow giữa $q^{(i)}$ và $q^{(i+1)}$ (chúng là queries khác nhau về ngữ nghĩa).

### 9.2. LoRA cập nhật xuyên iterations (within-instance)

Khác Level 1, LoRA **tích lũy trong một test instance** qua nhiều iterations. Điều này hợp lý vì:
- Tất cả iterations của một test instance đều phục vụ cùng một câu hỏi $q$
- Kiến thức từ hop 1 (tìm được gì) có thể hữu ích cho hop 2

Tổng gradient LoRA:

$$\Delta \theta_{LoRA} = \sum_{i=1}^{\eta_q} \nabla_{\theta_{LoRA}} \left(\mathcal{L}^{(i)}_{\text{LoRA}} + \beta \mathcal{L}_{\text{anchor}} + \gamma \mathcal{L}_{\text{LoRA-reg}}\right) \tag{32}$$

### 9.3. Vấn đề MHR tăng dần

ReSCORE được thiết kế để MHR$_i$@k **tăng dần** qua iterations. TTA phải không làm hỏng tính chất này. Đây là điều kiện cần kiểm tra thực nghiệm.

---

## 10. Thiết kế thực nghiệm

### 10.1. Research Questions

| RQ | Nội dung |
|---|---|
| **RQ1** | Dual pseudo-label tốt hơn CE-only hay LM-rel-only? |
| **RQ2** | Level 1 alone vs. Level 2 alone vs. kết hợp? |
| **RQ3** | Hard vs. Soft variant hiệu quả hơn trong MHQA setting? |
| **RQ4** | TTA có làm hỏng MHR$_i$@k tăng dần không? |
| **RQ5** | Anchor regularization strength $\lambda_1$ ảnh hưởng thế nào? |
| **RQ6** | LoRA rank $r$ tối ưu? |

### 10.2. OOD Evaluation Setup

**In-distribution:** Train ReSCORE trên MuSiQue, test trên MuSiQue (như paper gốc)  
**OOD Scenarios (3 loại):**

| Scenario | Train | Test OOD |
|---|---|---|
| Cross-dataset | MuSiQue | HotpotQA, 2WikiMHQA |
| Cross-dataset | HotpotQA | MuSiQue, 2WikiMHQA |
| Cross-hop | 2-hop (MuSiQue) | 3-hop (MuSiQue subset) |

### 10.3. Baselines

| Model | Mô tả |
|---|---|
| ReSCORE (no TTA) | Baseline chính |
| ReSCORE + TOUR$_{hard}$ (CE-only) | Level 1, hard variant |
| ReSCORE + TOUR$_{soft}$ (CE-only) | Level 1, soft variant |
| ReSCORE + TOUR$_{soft}$ (LM-rel-only) | Level 1, chỉ LM signal |
| **ReSCORE-TTA (Ours, L1)** | Level 1 với Dual pseudo-label |
| **ReSCORE-TTA (Ours, L1+L2)** | Level 1 + Level 2 (LoRA) |
| ReSCORE + BM25 (zero-shot) | Reference: sparse retrieval |

### 10.4. Metrics

- **QA:** EM, F1 (theo ReSCORE paper)
- **Retrieval:** MHR$_i$@k cho $i \in \{1, 2, \eta_n\}$, $k=8$ (theo ReSCORE paper)
- **Stability:** Variance across seeds; MHR không giảm dần
- **Efficiency:** Seconds/query, so sánh với ReSCORE no-TTA

### 10.5. Hyperparameters cần tuning

| Hyperparameter | Suggested range | Ghi chú |
|---|---|---|
| $T_{inner}$ (TOUR iterations) | 1–3 | Từ TOUR: max 3 |
| $\eta_q$ (query vector LR) | 0.5–2.0 | Từ TOUR: 1.2 cho DensePhrases |
| $\eta_{LoRA}$ (LoRA LR) | 1e-4 – 1e-3 | Standard LoRA |
| $r$ (LoRA rank) | 4, 8, 16 | |
| $\tau$ (CE temperature) | 0.1–1.0 | Từ TOUR: 0.5 |
| $p$ (nucleus threshold) | 0.5 | Từ TOUR: 0.5 |
| $\lambda_1$ (anchor weight) | 0.01–0.1 | |
| $\lambda_2$ (LoRA-reg weight) | 0.01–0.1 | |
| $\theta_{conf}$ (confidence filter) | 0.1–0.3 | |

---

## 11. Phân tích rủi ro (Devil's Advocate)

### 11.1. Rủi ro chính

**Risk 1: Dual pseudo-label vẫn nhiễu trên OOD**  
*Mô tả:* Cross-encoder được train trên in-distribution data; $P_{LM}(q \mid d)$ dùng LLM generalist. Trên một domain rất khác (e.g., medical MHQA), cả hai signal đều có thể unreliable.  
*Mức độ:* CAO  
*Mitigations:* Confidence filtering (Eq. 13); consistency check (Eq. 27)  

**Risk 2: Level 2 LoRA overfit vào noise**  
*Mô tả:* Với $\eta_q = 6$ iterations và noisy pseudo-labels, LoRA có thể học pattern sai trong một instance.  
*Mức độ:* TRUNG BÌNH  
*Mitigations:* Anchor regularization; LoRA-reg; có thể disable Level 2 hoàn toàn và chỉ dùng Level 1  

**Risk 3: Latency không chấp nhận được**  
*Mô tả:* Mỗi test instance cần: $\eta_q \times T_{inner}$ lần gradient computation + $\eta_q$ lần $P_{LM}(q \mid d)$ call.  
*Mức độ:* TRUNG BÌNH  
*TOUR baseline:* 0.44 s/query cho $k=10$. Với MHQA $\eta_q=6$: ~2.6s/query.  
*Mitigations:* Cache $P_{LM}(q \mid d_j)$ scores; early stopping; chỉ áp dụng TTA khi retrieval confidence thấp  

**Risk 4: TOUR's assumptions không phù hợp với iterative setting**  
*Mô tả:* TOUR được thiết kế cho single-hop QA với static corpus. Trong ReSCORE, query thay đổi tại mỗi iteration và documents retrieved ở hop $i$ bị loại trừ ở hop $i+1$ ($D^{(i)} \cap D^{(i+1)} = \emptyset$).  
*Mức độ:* TRUNG BÌNH  
*Mitigations:* Apply TOUR independently per-iteration (Eq. 31); không dùng inter-hop gradient flow  

**Risk 5: LLM call cho $P_{LM}(q \mid d)$ quá tốn kém**  
*Mô tả:* ReSCORE dùng top-M=32 documents; $P_{LM}(q \mid d)$ cần 32 LLM forward passes per iteration.  
*Mức độ:* CAO  
*Mitigations:* Batch processing; dùng smaller model cho relevance scoring; hoặc bỏ $P_{LM}(q \mid d)$ và chỉ dùng CE (degraded but faster)  

### 11.2. Bảng tóm tắt rủi ro

| Rủi ro | Mức độ | Mitigation | Fallback |
|---|---|---|---|
| Pseudo-label noise OOD | CAO | Confidence filtering + consensus | Chỉ dùng Level 1 + CE |
| LoRA overfit | TRUNG BÌNH | Anchor reg + LoRA-reg | Per-instance reset + low rank |
| Latency | TRUNG BÌNH | Cache + early stop | Chỉ Level 1 |
| TOUR-iterative mismatch | TRUNG BÌNH | Per-iteration independence | Không sao |
| $P_{LM}(q\|d)$ compute cost | CAO | Batch + small model | Drop LM signal |

---

## 12. Các hướng mở rộng tiếp theo

1. **Curriculum TTA:** Áp dụng TTA aggressively hơn khi confidence retriever thấp, nhẹ hơn khi đã confident
2. **Cross-instance LoRA bank:** Thay vì reset hoàn toàn, dùng một bank LoRA nhỏ theo domain clusters
3. **Thought-guided pseudo-label:** Dùng thought $t^{(i)}$ để augment query cho pseudo-label computation (Eq. 29)
4. **Learned temperature $\tau$:** Học $\tau$ tự động dựa trên domain distance indicator

---

## 13. Tóm tắt framework

```
ReSCORE-TTA = ReSCORE + TOUR-soft(Dual Pseudo-label) + Per-instance LoRA

Pseudo-GT tại test time: Q̃_TTA ∝ φ(q,d) · P_LM(q|d)  [không cần answer a]

Level 1 (zero forgetting):   optimize query vector q_t via TOUR-soft
Level 2 (bounded forgetting): optimize LoRA(query encoder), reset per-instance

Loss: L_TTA = L_soft (Level 1) + α·L_LoRA (Level 2)
            + β·L_anchor (anti-forgetting) + γ·L_LoRA-reg (LoRA bound)
```

---

*Tài liệu này dựa hoàn toàn trên nội dung của ReSCORE (ACL 2025) và TOUR (ACL 2023 Findings). Các đề xuất được phân loại rõ ràng là FACT / INFERENCE / HYPOTHESIS theo research protocol.*
