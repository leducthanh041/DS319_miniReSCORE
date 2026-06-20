"""
cross_encoder_wrapper.py
========================
Passage-level cross-encoder wrapper cho TTA pseudo-labels trong ReSCORE-TTA.

Adapted từ TOUR cross_encoder.py (phrase-level) → đơn giản hơn vì ReSCORE
làm việc với passages (không phải phrases), nên không cần 3-sent windowing
hay [S][E] phrase tagging.

Chức năng chính:
    score_documents(query, docs)   → raw relevance logits [N]
    get_soft_labels(...)           → softmax distribution [N]   (dùng cho TOUR-soft)
    get_hard_labels(...)           → nucleus-selected indices   (dùng cho TOUR-hard)

Cache: kết quả cross-encoder được cache để tránh re-computation khi
cùng một (query, doc) pair xuất hiện nhiều lần trong T_inner steps.
Tương đương CE_Cache trong TOUR §3.5.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional, Tuple
import numpy as np


class CrossEncoderWrapper:
    """
    Wrapper cho HuggingFace cross-encoder để score passage-query pairs.

    Khuyến nghị model:
        - 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (nhỏ, nhanh)
        - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (lớn hơn, chính xác hơn)
        - 'cross-encoder/ms-marco-electra-base'    (mạnh nhất, chậm nhất)

    Args:
        model_name_or_path: HuggingFace model identifier hoặc local path.
        device: 'cuda', 'cpu', hoặc 'cuda:N'. None → auto-detect.
        max_length: max token length cho cross-encoder input.
        batch_size: batch size khi scoring nhiều documents cùng lúc.
    """

    def __init__(
        self,
        model_name_or_path: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[CrossEncoderWrapper] Loading {model_name_or_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)
        self.model.eval()

        # Cache: (query_prefix, doc_prefix) → float score
        # Tương đương CE_Cache trong TOUR để tránh redundant computation
        self._cache: dict = {}

    # ──────────────────────────────────────────────────────────────
    # Cache helpers
    # ──────────────────────────────────────────────────────────────

    def _cache_key(self, query: str, doc_text: str) -> str:
        # Dùng prefix để giới hạn key size
        return f"{query[:120]}|||{doc_text[:250]}"

    def _get_cached(self, query: str, doc_text: str) -> Optional[float]:
        return self._cache.get(self._cache_key(query, doc_text))

    def _set_cached(self, query: str, doc_text: str, score: float):
        self._cache[self._cache_key(query, doc_text)] = score

    def clear_cache(self):
        """Xóa cache. Gọi giữa các test instances để tiết kiệm RAM."""
        self._cache.clear()

    # ──────────────────────────────────────────────────────────────
    # Core scoring
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_documents(
        self,
        query: str,
        documents: List[str],
    ) -> torch.Tensor:
        """
        Score một query với nhiều documents bằng cross-encoder.

        Model output logit được xử lý:
        - Nếu num_labels == 2: lấy logits[:, 1] (positive class score)
        - Nếu num_labels == 1: lấy logits[:, 0] (regression score)

        Returns:
            scores: Tensor shape [N], dtype float32.
                    Score cao hơn = document relevant hơn với query.

        Note:
            Kết quả được cache tự động. Lần gọi sau với cùng (query, doc)
            sẽ dùng cache thay vì chạy model lại.
        """
        if len(documents) == 0:
            return torch.tensor([], dtype=torch.float32)

        scores_list: List[Optional[float]] = [None] * len(documents)
        uncached_indices: List[int] = []
        uncached_docs: List[str] = []

        # Phân loại: cached vs uncached
        for i, doc in enumerate(documents):
            cached_score = self._get_cached(query, doc)
            if cached_score is not None:
                scores_list[i] = cached_score
            else:
                uncached_indices.append(i)
                uncached_docs.append(doc)

        # Batch inference cho uncached documents
        if uncached_docs:
            computed_scores = self._batch_score(query, uncached_docs)
            for local_idx, (orig_idx, doc) in enumerate(
                zip(uncached_indices, uncached_docs)
            ):
                score_val = computed_scores[local_idx].item()
                self._set_cached(query, doc, score_val)
                scores_list[orig_idx] = score_val

        return torch.tensor(scores_list, dtype=torch.float32)

    def _batch_score(
        self,
        query: str,
        documents: List[str],
    ) -> torch.Tensor:
        """Chạy cross-encoder inference trên một list documents (không dùng cache)."""
        all_scores: List[float] = []

        for batch_start in range(0, len(documents), self.batch_size):
            batch_docs = documents[batch_start: batch_start + self.batch_size]
            queries_batch = [query] * len(batch_docs)

            encodings = self.tokenizer(
                queries_batch,
                batch_docs,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors='pt',
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                logits = self.model(**encodings).logits  # [B, num_labels]

            # Lấy relevance score tùy theo số output classes của model
            if logits.shape[-1] == 2:
                batch_scores = logits[:, 1]   # positive class
            else:
                batch_scores = logits[:, 0]   # single regression output

            all_scores.extend(batch_scores.cpu().tolist())

        return torch.tensor(all_scores, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────────
    # Pseudo-label generation
    # ──────────────────────────────────────────────────────────────

    def get_soft_labels(
        self,
        query: str,
        documents: List[str],
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Soft pseudo-labels: softmax của CE scores với temperature tau.

        Tương đương P(c | q, phi) trong TOUR Eq. (8) / Eq. (12):
            P(c_i | q, phi) = softmax(phi(q, c_i) / tau)

        Args:
            query: câu truy vấn
            documents: list document texts
            tau: temperature parameter (TOUR dùng 0.5)

        Returns:
            Tensor [N] — probability distribution, sum = 1.
        """
        raw_scores = self.score_documents(query, documents)
        return F.softmax(raw_scores / tau, dim=0)

    def get_hard_labels(
        self,
        query: str,
        documents: List[str],
        tau: float = 0.5,
        p: float = 0.5,
    ) -> List[int]:
        """
        Hard pseudo-labels: nucleus selection.

        Tương đương C^q_hard trong TOUR Eq. (4b):
            Chọn tập nhỏ nhất S sao cho sum_{c in S} P(c|q,phi) >= p

        Giống Nucleus Sampling (Holtzman et al., 2020) được TOUR áp dụng
        để chọn pseudo-positive documents.

        Args:
            query: câu truy vấn
            documents: list document texts
            tau: temperature parameter
            p: nucleus threshold (TOUR dùng 0.5)

        Returns:
            List[int] — indices của pseudo-positive documents trong `documents`.
        """
        raw_scores = self.score_documents(query, documents)
        probs = F.softmax(raw_scores / tau, dim=0)  # [N]

        # Sort descending by probability
        sorted_idx = torch.argsort(probs, descending=True)
        cumsum = torch.cumsum(probs[sorted_idx], dim=0)

        # Tập nhỏ nhất để cumsum >= p
        exceed = (cumsum >= p).nonzero(as_tuple=True)[0]
        if len(exceed) == 0:
            # Không đạt threshold với bất kỳ prefix nào → chọn tất cả
            n_select = len(documents)
        else:
            n_select = exceed[0].item() + 1

        return sorted_idx[:n_select].tolist()

    # ──────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CrossEncoderWrapper("
            f"model={self.model_name_or_path}, "
            f"device={self.device}, "
            f"cache_size={len(self._cache)})"
        )
