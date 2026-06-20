"""
tta_retrieval.py
================
TTARetrievalStep: Thay thế RetrievalStep trong inference pipeline
để thực hiện Test-Time Adaptation.

Flow cho mỗi RAG iteration i:
  1. Reset LoRA nếu đây là instance mới (Level 2)
  2. Tính query embedding q_0 từ encoder (có LoRA nếu Level 2)
  3. Initial retrieval top-M documents (ANN search)
  4. Tính Dual Pseudo-label Q_TTA từ (CE + LM_rel scores)
  5. Level 1: Optimize query vector q_t qua T_inner gradient steps
  6. Level 2: Update LoRA params qua 1 gradient step
  7. Final retrieval với optimized q_T
  8. Return DocumentState với top-k documents

Về batching và Level 2:
  - Level 1 (query vector): hoàn toàn per-instance, safe với batch_size > 1
  - Level 2 (LoRA): LoRA là shared weight → cần batch_size=1 để đảm bảo
    per-instance isolation. Nếu batch_size > 1, LoRA sẽ mix gradient của
    nhiều instances trong cùng batch, gây nhiễu.
  - inference_tta.py tự động force batch_size=1 khi tta_level in ('l2', 'both')

References:
  - TOUR §3.3 (TOURsoft): query vector optimization với KL loss
  - TOUR §3.5 (Efficient implementation): early stopping, caching
  - ReSCORE §3.2: KL divergence training structure
  - ReSCORE-TTA framework: Dual pseudo-label (Eq. 11-12)
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from transformers import get_linear_schedule_with_warmup

from source.module.index.docstore import Document
from source.module.retrieve.lora_utils import (
    get_lora_parameters,
    lora_disabled,
    lora_norm_regularization,
    reset_lora,
)
from source.pipeline.constants import THOUGHT_THOUGHT_DELIM
from source.pipeline.state import BaseState, DocumentState
from source.pipeline.utils import (
    filter_document,
    parse_path,
    preprocess_retrieval_query,
)


class TTARetrievalStep:
    """
    Drop-in replacement cho RetrievalStep trong inference pipeline.

    Hỗ trợ 3 modes:
        tta_level='l1':   chỉ query vector optimization (TOUR-style)
        tta_level='l2':   chỉ LoRA adaptation (cần batch_size=1)
        tta_level='both': cả hai (cần batch_size=1 vì Level 2)

    Và 3 pseudo-label variants:
        tta_pseudo_label='ce_only':  chỉ cross-encoder (nhanh nhất)
        tta_pseudo_label='lm_only':  chỉ P_LM(q|d) (không cần CE)
        tta_pseudo_label='dual':     CE * P_LM(q|d) (tốt nhất, chậm nhất)

    Args:
        cfg: PipelineConfig với các TTA fields đã được set.
        retriever: DenseRetriever đã có LoRA injected (nếu Level 2).
        indexer: Indexer với FAISS index.
        cross_encoder: CrossEncoderWrapper instance.
        generator: LlamaGenerator để tính P_LM(q|d) (cho dual pseudo-label).
        retrieval_trace_file_path: path để ghi retrieval trace (optional).
    """

    def __init__(
        self,
        cfg,
        retriever,
        indexer,
        cross_encoder,
        generator,
        retrieval_trace_file_path: Optional[str] = None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.indexer = indexer
        self.cross_encoder = cross_encoder
        self.generator = generator
        self.retrieval_trace_file_path = retrieval_trace_file_path

        # Load prompt templates cho P_LM(q|d) — chỉ dùng khi dual/lm_only
        self._q_rel_input_template: Optional[str] = None
        self._q_rel_output_template: Optional[str] = None

        if cfg.tta_pseudo_label in ('dual', 'lm_only'):
            if os.path.exists(cfg.tta_q_rel_input_prompt_file_path):
                with open(cfg.tta_q_rel_input_prompt_file_path, 'r', encoding='utf-8') as f:
                    self._q_rel_input_template = f.read()
                with open(cfg.tta_q_rel_output_prompt_file_path, 'r', encoding='utf-8') as f:
                    self._q_rel_output_template = f.read()
            else:
                print(
                    f"[TTARetrievalStep] WARNING: q_rel prompt not found at "
                    f"{cfg.tta_q_rel_input_prompt_file_path}. "
                    f"Falling back to ce_only pseudo-label."
                )
                self.cfg.tta_pseudo_label = 'ce_only'

        # Per-instance LoRA state tracking
        # set of question_ids đã được reset LoRA trong session này
        self._lora_reset_done: set = set()

        # LoRA optimizer — được tạo mới cho mỗi instance
        self._lora_optimizer: Optional[torch.optim.Optimizer] = None
        self._trace_count = 0
        self._lm_relevance_cache: Dict[Tuple[str, str], float] = {}

    # ══════════════════════════════════════════════════════════════
    # PUBLIC: Main call (gọi bởi PipelineController)
    # ══════════════════════════════════════════════════════════════

    def __call__(
        self,
        paths: List[List[BaseState]],
    ) -> List[BaseState]:
        """
        Được gọi bởi PipelineController tại mỗi retrieval turn.

        Args:
            paths: list of state paths. Mỗi path là history của 1 instance.

        Returns:
            list of DocumentState — mỗi state chứa top-k documents đã retrieve.
        """
        all_next_states: List[DocumentState] = []
        trace_records: List[Dict[str, Any]] = []

        for path in paths:
            question_id, question, thoughts, prev_documents = parse_path(path)
            iteration = len(prev_documents) + 1
            document_ids_so_far = {doc.id for doc in sum(prev_documents, [])}

            if (
                iteration == 1
                and getattr(self.cfg, 'tta_clear_cross_encoder_cache', True)
                and self.cross_encoder is not None
            ):
                self.cross_encoder.clear_cache()
            if iteration == 1:
                self._lm_relevance_cache.clear()

            # ── 0. Reset LoRA nếu đây là instance mới ──────────────
            if self.cfg.tta_level in ('l2', 'both'):
                if question_id not in self._lora_reset_done:
                    n_reset = reset_lora(self.retriever.query_model)
                    self._lora_reset_done.add(question_id)
                    # Tạo optimizer LoRA mới cho instance này
                    lora_params = get_lora_parameters(self.retriever.query_model)
                    if lora_params:
                        self._lora_optimizer = Adam(
                            lora_params,
                            lr=self.cfg.tta_lora_lr,
                        )

            # ── 1. Query string ─────────────────────────────────────
            query_str = preprocess_retrieval_query(
                question,
                thoughts,
                retrieval_query_type=self.cfg.retrieval_query_type,
            )

            # ── 2. Query embedding q_0 with current within-instance LoRA ──
            with torch.no_grad():
                q0_for_search = self.retriever.embed_single_query_no_detach(
                    query_str
                ).detach().float()

            # Keep the initial ranking for trace/debug comparisons.
            retrieved_docs_m, _ = self._retrieve_candidates(q0_for_search)

            # ── 3. Level 1: TOUR query-vector optimization ─────────
            q_final = q0_for_search.clone()
            l1_stats: Dict[str, Any] = {
                'enabled': self.cfg.tta_level in ('l1', 'both'),
                'steps': 0,
                'early_stopped': False,
                'pseudo_label_failures': 0,
            }

            if self.cfg.tta_level in ('l1', 'both'):
                q_final, l1_stats = self._optimize_query_vector_l1(
                    query_str=query_str,
                    q0=q0_for_search,
                )

            # ── 4. Level 2: per-instance LoRA update ───────────────
            l2_stats: Dict[str, Any] = {
                'enabled': self.cfg.tta_level in ('l2', 'both'),
                'updated': False,
            }
            pseudo_label_ok = l1_stats.get('pseudo_label_failures', 0) == 0
            if (
                self.cfg.tta_level in ('l2', 'both')
                and self._lora_optimizer is not None
            ):
                l2_docs, l2_doc_embeddings = self._retrieve_candidates(q_final)
                l2_pseudo_gt = self._compute_pseudo_label(
                    query_str=query_str,
                    doc_texts=[_doc_to_text(doc) for doc in l2_docs],
                    doc_embeddings=l2_doc_embeddings,
                    q_current=q_final,
                )
                pseudo_label_ok = pseudo_label_ok and l2_pseudo_gt is not None
                l1_delta = q_final - q0_for_search
                if l2_pseudo_gt is not None:
                    with torch.no_grad(), lora_disabled(
                        self.retriever.query_model
                    ):
                        q_base = self.retriever.embed_single_query_no_detach(
                            query_str
                        ).detach().float()
                    l2_stats = self._update_lora_l2(
                        query_str=query_str,
                        pseudo_gt=l2_pseudo_gt,
                        doc_embeddings=l2_doc_embeddings,
                        q_base_anchor=q_base,
                    )
                    with torch.no_grad():
                        q_after_lora = self.retriever.embed_single_query_no_detach(
                            query_str
                        ).detach().float()

                    # q_TOUR = q_0 + delta_TOUR. After LoRA changes q_0,
                    # preserve the instance-level TOUR delta on the new base.
                    q_final = (
                        q_after_lora + l1_delta
                        if self.cfg.tta_level == 'both'
                        else q_after_lora
                    )

            # ── 5. Final retrieval with adapted query ───────────────
            q_final_np = q_final.detach().cpu().numpy()[np.newaxis].astype('float32')
            indexer_out_k = self.indexer.search(
                query_embeddings=q_final_np,
                k=self.cfg.retrieval_buffer_size,
            )[0]

            documents = filter_document(
                documents=indexer_out_k.documents,
                document_ids_so_far=document_ids_so_far,
                retrieval_no_duplicates=self.cfg.retrieval_no_duplicates,
            )

            # ── 8. Build output state ───────────────────────────────
            all_next_states.append(
                DocumentState(
                    parent_state_id=path[-1].state_id,
                    documents=documents,
                    question_id=question_id,
                    question=question,
                )
            )

            trace_records.append({
                'question_id': question_id,
                'iteration': iteration,
                'query': query_str,
                'tta_level': self.cfg.tta_level,
                'tta_pseudo_label': self.cfg.tta_pseudo_label,
                'pseudo_label_ok': pseudo_label_ok,
                'retrieval_count': self.cfg.retrieval_count,
                'retrieval_buffer_size': self.cfg.retrieval_buffer_size,
                'adaptation': {
                    'query_shift_l2': float(
                        torch.linalg.vector_norm(q_final - q0_for_search).item()
                    ),
                    'l1': l1_stats,
                    'l2': l2_stats,
                },
                'initial_documents': [
                    _document_to_trace(doc, rank=r)
                    for r, doc in enumerate(
                        retrieved_docs_m[:self.cfg.retrieval_count], 1
                    )
                ],
                'documents': [
                    _document_to_trace(doc, rank=r)
                    for r, doc in enumerate(
                        documents[:self.cfg.retrieval_count], 1
                    )
                ],
            })

            self._trace_count += 1
            log_every = max(1, int(getattr(self.cfg, 'tta_log_every', 1)))
            if self._trace_count % log_every == 0:
                print(
                    f"[TTA] qid={question_id} hop={iteration} "
                    f"pseudo_ok={pseudo_label_ok} "
                    f"l1_steps={l1_stats.get('steps', 0)} "
                    f"l1_loss={l1_stats.get('final_loss', 'n/a')} "
                    f"query_shift={trace_records[-1]['adaptation']['query_shift_l2']:.6f} "
                    f"l2_loss={l2_stats.get('loss', 'n/a')}"
                )

        self._append_trace(trace_records)
        return all_next_states

    def _retrieve_candidates(
        self,
        query_vector: torch.Tensor,
    ) -> Tuple[List[Document], torch.Tensor]:
        query_np = query_vector.detach().cpu().numpy()[np.newaxis].astype('float32')
        indexer_output = self.indexer.search(
            query_embeddings=query_np,
            k=self.cfg.retrieval_buffer_size,
        )[0]
        documents = indexer_output.documents
        if not documents:
            raise RuntimeError("TTA retrieval returned no candidate documents.")

        embeddings = torch.stack([
            self.indexer.get_embedding_from_docstore_id(
                document.id,
                return_tensors='pt',
            ).to(query_vector.device)
            for document in documents
        ], dim=0).float()
        return documents, embeddings

    # ══════════════════════════════════════════════════════════════
    # PRIVATE: Pseudo-label computation
    # ══════════════════════════════════════════════════════════════

    def _compute_pseudo_label(
        self,
        query_str: str,
        doc_texts: List[str],
        doc_embeddings: torch.Tensor,
        q_current: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Tính Dual Pseudo-label Q_TTA (Eq. 12 trong research doc).

        Q_TTA ∝ P_CE(d|q) · P_LM(q|d)

        Các phân nhánh:
          ce_only:  Q_TTA = softmax(phi(q,d) / tau)
          lm_only:  Q_TTA = softmax(log P_LM(q|d) / tau)
          dual:     Q_TTA ∝ softmax(phi(q,d)/tau) * exp(-ppl(q|d))

        Returns:
            Tensor [M] — normalized probability distribution, hoặc None nếu lỗi.
        """
        try:
            if self.cfg.tta_pseudo_label == 'ce_only':
                # Option A: Cross-encoder only (TOUR approach)
                # Tương đương P(c|q,phi) trong TOUR Eq. (8)
                ce_scores = self.cross_encoder.score_documents(query_str, doc_texts)
                pseudo_gt = F.softmax(
                    ce_scores.to(q_current.device) / self.cfg.tta_temperature, dim=0
                )

            elif self.cfg.tta_pseudo_label == 'lm_only':
                # Option B: LM relevance only — P_LM(q|d) (no answer needed)
                # Từ ReSCORE Table 3: +5.37% recall so với baseline
                lm_rel = self._compute_lm_relevance(query_str, doc_texts)
                lm_rel = lm_rel.to(q_current.device).clamp(min=1e-9)
                pseudo_gt = lm_rel / lm_rel.sum()

            else:  # 'dual' (default, recommended)
                # Option D: Dual pseudo-label — kết hợp CE + LM_rel
                # Q_TTA ∝ CE_prob(d|q) * P_LM(q|d)
                ce_scores = self.cross_encoder.score_documents(query_str, doc_texts)
                ce_probs = F.softmax(
                    ce_scores.to(q_current.device) / self.cfg.tta_temperature, dim=0
                )  # [M]

                lm_rel = self._compute_lm_relevance(query_str, doc_texts)
                lm_rel = lm_rel.to(q_current.device).clamp(min=1e-9)

                # Kết hợp: element-wise product rồi normalize
                combined = ce_probs * lm_rel
                if combined.sum() < 1e-9:
                    # Fallback nếu combined quá nhỏ
                    pseudo_gt = ce_probs
                else:
                    pseudo_gt = combined / combined.sum()

            confidence_threshold = self.cfg.tta_confidence_threshold
            if confidence_threshold > 0:
                confidence_mask = pseudo_gt >= confidence_threshold
                if not torch.any(confidence_mask):
                    confidence_mask[torch.argmax(pseudo_gt)] = True
                pseudo_gt = pseudo_gt * confidence_mask
                pseudo_gt = pseudo_gt / pseudo_gt.sum().clamp(min=1e-12)

            return pseudo_gt.detach()

        except Exception as exc:
            message = f"TTA pseudo-label computation failed: {exc}"
            if self.cfg.tta_fail_on_pseudo_label_error:
                raise RuntimeError(message) from exc
            print(f"[TTARetrievalStep] {message}")
            return None

    def _compute_lm_relevance(
        self,
        query_str: str,
        doc_texts: List[str],
    ) -> torch.Tensor:
        """
        Tính P_LM(q|d) cho mỗi document.

        Dùng generator.score(condition_prompt, output) để tính perplexity
        của câu hỏi query_str khi cho trước document.
        P_LM(q|d) ∝ exp(-perplexity) → score cao = perplexity thấp = relevant.

        Args:
            query_str: câu truy vấn (dùng làm "answer" trong scoring)
            doc_texts: list document texts (dùng làm condition)

        Returns:
            Tensor [M] — unnormalized P_LM(q|d) proxies (all positive).
        """
        keys = [(query_str, doc_text) for doc_text in doc_texts]
        missing_keys = [key for key in keys if key not in self._lm_relevance_cache]
        if missing_keys:
            prompts = [
                self._q_rel_input_template.format(documents=doc_text)
                for _, doc_text in missing_keys
            ]
            outputs = [
                self._q_rel_output_template.format(question=query)
                for query, _ in missing_keys
            ]
            perplexities: List[float] = self.generator.score(prompts, outputs)
            if len(perplexities) != len(missing_keys):
                raise RuntimeError(
                    "Generator returned an unexpected number of LM relevance scores."
                )
            for key, perplexity in zip(missing_keys, perplexities):
                # score() returns exp(mean NLL), so exp(-mean NLL) = 1 / PPL.
                self._lm_relevance_cache[key] = 1.0 / max(float(perplexity), 1e-6)

        return torch.tensor(
            [self._lm_relevance_cache[key] for key in keys],
            dtype=torch.float32,
        )

    # ══════════════════════════════════════════════════════════════
    # PRIVATE: Level 1 — Query Vector Optimization
    # ══════════════════════════════════════════════════════════════

    def _optimize_query_vector_l1(
        self,
        query_str: str,
        q0: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Level 1: Optimize query vector q_t bằng SGD với T_inner steps.

        Tương đương TOUR-soft (Eq. 16-18) với teacher = Q_TTA:
            L_soft = KL(Q_TTA || P_R(·|q_t))
            q_{t+1} = q_t - eta * grad(L_soft) - beta * (q_t - q0) * 2

        Update rule implicit (từ TOUR Appendix B, Eq. 8):
            q_{t+1} = q_t + eta * [sum_j Q_TTA(j)*d_j - sum_j P_R(j|q_t)*d_j]
                    - beta * 2 * (q_t - q0)   ← anchor gradient

        Args:
            q0: initial query vector [d] (output của encoder, detached)
            pseudo_gt: Q_TTA distribution [M]
            doc_embeddings: document vectors [M, d]

        Returns:
            q_T: optimized query vector [d], detached.
        """
        # q_t là leaf tensor được optimize
        q_t = q0.clone().requires_grad_(True)

        optimizer = SGD(
            [q_t],
            lr=self.cfg.tta_query_lr,
            momentum=self.cfg.tta_momentum,
            weight_decay=self.cfg.tta_weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(
                self.cfg.tta_warmup_steps,
                self.cfg.tta_inner_steps,
            ),
            num_training_steps=max(1, self.cfg.tta_inner_steps),
        )

        stats: Dict[str, Any] = {
            'enabled': True,
            'steps': 0,
            'early_stopped': False,
            'pseudo_label_failures': 0,
            'candidate_refreshes': 0,
        }
        fixed_candidates = None

        for step in range(self.cfg.tta_inner_steps):
            if (
                fixed_candidates is None
                or self.cfg.tta_refresh_candidates_each_step
            ):
                fixed_candidates = self._retrieve_candidates(q_t.detach())
                stats['candidate_refreshes'] += 1
            documents, doc_embeddings = fixed_candidates
            pseudo_gt = self._compute_pseudo_label(
                query_str=query_str,
                doc_texts=[_doc_to_text(document) for document in documents],
                doc_embeddings=doc_embeddings,
                q_current=q_t.detach(),
            )
            if pseudo_gt is None:
                stats['pseudo_label_failures'] += 1
                break

            positive_indices = self._nucleus_positive_indices(
                pseudo_gt,
                self.cfg.tta_nucleus_p,
            )
            if 0 in positive_indices or not positive_indices:
                stats['early_stopped'] = True
                stats['early_stop_inner_step'] = step
                break

            optimizer.zero_grad()

            # Retriever distribution P_R(d|q_t) = softmax(q_t @ D^T / tau)
            # [M] — normalized over top-M candidates
            logits = q_t @ doc_embeddings.T  # [M]
            log_p_retriever = F.log_softmax(
                logits / self.cfg.tta_temperature, dim=0
            )  # [M]

            # TOUR-soft KL loss: KL(Q_TTA || P_R)
            # F.kl_div(input=log_P, target=Q) = sum_i Q_i * (log Q_i - log P_i)
            loss_kl = F.kl_div(
                log_p_retriever,
                pseudo_gt.detach(),
                reduction='sum',
            )

            # Anchor regularization: beta * ||q_t - q_0||^2
            # Ngăn q_t lệch quá xa khỏi pretrained embedding
            loss_anchor = self.cfg.tta_anchor_weight * (
                q_t - q0.detach()
            ).pow(2).sum()

            loss = loss_kl + loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [q_t],
                max_norm=self.cfg.tta_max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            stats.update({
                'steps': step + 1,
                'final_loss': float(loss.detach().item()),
                'final_kl_loss': float(loss_kl.detach().item()),
                'final_anchor_loss': float(loss_anchor.detach().item()),
                'learning_rate': float(scheduler.get_last_lr()[0]),
                'nucleus_positive_count': len(positive_indices),
            })

        return q_t.detach(), stats

    @staticmethod
    def _nucleus_positive_indices(
        pseudo_gt: torch.Tensor,
        nucleus_p: float,
    ) -> set:
        sorted_indices = torch.argsort(pseudo_gt, descending=True)
        cumulative = torch.cumsum(pseudo_gt[sorted_indices], dim=0)
        threshold_positions = torch.where(cumulative >= nucleus_p)[0]
        cutoff = (
            int(threshold_positions[0].item()) + 1
            if len(threshold_positions) > 0
            else len(sorted_indices)
        )
        return set(sorted_indices[:cutoff].tolist())

    # ══════════════════════════════════════════════════════════════
    # PRIVATE: Level 2 — LoRA Update
    # ══════════════════════════════════════════════════════════════

    def _update_lora_l2(
        self,
        query_str: str,
        pseudo_gt: torch.Tensor,       # [M]
        doc_embeddings: torch.Tensor,  # [M, d]
        q_base_anchor: torch.Tensor,    # [d], frozen encoder without LoRA
    ) -> Dict[str, Any]:
        """
        Level 2: 1 gradient step để update LoRA parameters.

        Loss = alpha * KL + beta * ||q_lora-q_base||^2 + gamma * ||BA||_F^2

        Cấu trúc loss y hệt ReSCORE Eq. (2) nhưng:
        - Teacher: Q_TTA thay vì Q_LM (không cần answer a)
        - Student params: chỉ LoRA {B, A} thay vì toàn bộ encoder

        LoRA KHÔNG bị reset sau mỗi hop — tích lũy trong suốt một instance
        (tất cả iterations của cùng một câu hỏi).
        LoRA BỊ reset tại đầu instance mới (trong __call__).
        """
        self.retriever.query_model.train()
        self._lora_optimizer.zero_grad()

        # Forward qua encoder với LoRA active — giữ gradient
        q_lora = self.retriever.embed_single_query_no_detach(query_str)  # [d]

        # Retriever distribution với LoRA params
        logits = q_lora @ doc_embeddings.T  # [M]
        log_p_lora = F.log_softmax(
            logits / self.cfg.tta_temperature, dim=0
        )  # [M]

        # KL divergence loss
        loss_kl = F.kl_div(
            log_p_lora,
            pseudo_gt.detach(),
            reduction='sum',
        )

        loss_anchor = self.cfg.tta_anchor_weight * (
            q_lora - q_base_anchor.detach()
        ).pow(2).sum()

        # LoRA norm regularization: gamma * ||BA||_F^2
        # Giới hạn magnitude của LoRA update
        loss_reg = self.cfg.tta_lora_reg_weight * lora_norm_regularization(
            self.retriever.query_model
        )

        weighted_kl = self.cfg.tta_lora_loss_weight * loss_kl
        loss = weighted_kl + loss_anchor + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            get_lora_parameters(self.retriever.query_model),
            max_norm=self.cfg.tta_max_grad_norm,
        )
        self._lora_optimizer.step()

        self.retriever.query_model.eval()
        return {
            'enabled': True,
            'updated': True,
            'loss': float(loss.detach().item()),
            'kl_loss': float(loss_kl.detach().item()),
            'weighted_kl_loss': float(weighted_kl.detach().item()),
            'anchor_loss': float(loss_anchor.detach().item()),
            'regularization_loss': float(loss_reg.detach().item()),
        }

    # ══════════════════════════════════════════════════════════════
    # PRIVATE: Trace logging
    # ══════════════════════════════════════════════════════════════

    def _append_trace(self, records: List[Dict[str, Any]]):
        if not self.retrieval_trace_file_path or not records:
            return
        os.makedirs(os.path.dirname(self.retrieval_trace_file_path), exist_ok=True)
        with open(self.retrieval_trace_file_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ──────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────

def _doc_to_text(doc: Document) -> str:
    """Convert Document object thành string cho cross-encoder / LM scoring."""
    if doc.metadata:
        title = doc.metadata.get('title', '')
        text = doc.metadata.get('text', '') or doc.content or ''
        if title:
            return f"{title} {text}".strip()
        return text.strip()
    return (doc.content or '').strip()


def _document_to_trace(doc: Document, rank: int) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        'rank': rank,
        'id': doc.id,
        'title': metadata.get('title'),
        'paragraph_text': metadata.get('text') or doc.content,
    }
