"""
dense_patch.py
==============
Patch này chỉ chứa METHOD MỚI cần thêm vào class DenseRetriever
trong file source/module/retrieve/dense.py.

HƯỚNG DẪN ÁP DỤNG:
    Mở source/module/retrieve/dense.py, tìm cuối class DenseRetriever
    (sau method _embed_queries), và paste method embed_single_query_no_detach
    vào trong class.

KHÔNG thay đổi bất kỳ code nào khác trong dense.py.
"""

# ── PASTE ĐOẠN NÀY VÀO CUỐI CLASS DenseRetriever (trước def pooling) ──

"""
    def embed_single_query_no_detach(
        self,
        query_text: str,
    ) -> torch.Tensor:
        \"\"\"
        Embed một query duy nhất và GIỮ NGUYÊN gradient computation graph.

        Khác với embed() vì:
        - embed() chạy batched, có torch.no_grad() trong inference mode
        - Method này KHÔNG dùng torch.no_grad() → gradient flow qua output
        - Chỉ nhận 1 string (không batched)

        Dùng cho TTA:
        - Level 1: lấy q_0 = E_q(query) làm điểm khởi đầu, sau đó
                   detach + requires_grad_(True) để optimize q_t độc lập
        - Level 2: giữ gradient để backprop qua LoRA parameters

        Args:
            query_text: một câu truy vấn dưới dạng string.

        Returns:
            Tensor shape [d] (embedding dimension).
            - Nếu query_model.training=True: có gradient graph (cho Level 2)
            - Nếu dùng q0.detach().clone().requires_grad_(True): cho Level 1
        \"\"\"
        model_inputs = self.query_tokenizer.batch_encode_plus(
            [query_text],
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=True,
            truncation=True,
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Không wrap trong torch.no_grad() — cần gradient để:
        # - Level 2: backprop qua LoRA params (B, A matrices)
        # - Level 1: lấy q_0 rồi detach, nên không cần grad ở đây,
        #            nhưng giữ chung interface cho cả hai level
        model_outputs = self.query_model(**model_inputs)

        embedding = pooling(
            token_embeddings=model_outputs[0],
            mask=model_inputs['attention_mask'],
            pooling=self.cfg.pooling,
            normalize=self.cfg.normalize,
        )

        return embedding[0]  # [d] — remove batch dimension
"""

# ── KẾT THÚC ĐOẠN PASTE ──


# Để verify sau khi patch:
# from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
# import torch
#
# retriever = DenseRetriever(DenseRetrieverConfig(training_strategy='query_only'))
# emb = retriever.embed_single_query_no_detach("What is multi-hop QA?")
# assert emb.shape == (768,), f"Expected (768,), got {emb.shape}"
# print(f"[OK] embed_single_query_no_detach output shape: {emb.shape}")
#
# # Test Level 1 usage pattern:
# q0 = emb.detach().clone().requires_grad_(True)
# assert q0.requires_grad, "q0 should require grad for L1 optimization"
# print(f"[OK] q0 requires_grad: {q0.requires_grad}")
