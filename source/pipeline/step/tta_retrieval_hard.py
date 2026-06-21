"""
tta_retrieval_hard.py
=====================
TOUR_hard variant của TTARetrievalStep.

Khác tta_retrieval.py (TOUR_soft) chỉ ở MỘT ĐIỂM:
    Hàm loss trong _optimize_query_vector_l1():

    TOUR_soft (tta_retrieval.py):
        L_soft = KL(Q_TTA || P_R) = -Σ Q_TTA(j) * log P_R(j|q_t)

    TOUR_hard (file này):
        L_hard = -log Σ_{c̃ ∈ C_hard} P_k(c̃|q_t)

    C_hard = nucleus-selected pseudo-positives từ Q_TTA:
        Tập nhỏ nhất s.t. Σ_{c̃ ∈ S} Q_TTA(c̃) >= p

Toàn bộ phần còn lại — pseudo-label, LoRA, early stopping, logging —
giữ nguyên từ TTARetrievalStep (kế thừa, không copy).

References:
    TOUR §3.2 (TOUR_hard): L_hard = -log Σ_{c̃ ∈ C_hard} P_k(c̃|q), Eq. (9)
    TOUR §3.5: early stopping khi c1 ∈ C^{q_t}_{hard}
    TOUR Table 2: TOUR_hard tốt hơn TOUR_soft trong OOD setting
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import SGD
from transformers import get_linear_schedule_with_warmup

from source.pipeline.step.tta_retrieval import TTARetrievalStep, _doc_to_text


class TTARetrievalStepHard(TTARetrievalStep):
    """
    TOUR_hard variant: dùng Maximum Marginal Likelihood loss trên
    nucleus-selected pseudo-positives thay vì KL divergence.

    Kế thừa toàn bộ từ TTARetrievalStep, chỉ override:
        _optimize_query_vector_l1()

    Tất cả các method khác (pseudo-label, LoRA, trace) giữ nguyên.

    Khi nào dùng TOUR_hard thay TOUR_soft?
        - TOUR paper (Table 2): TOUR_hard consistently better trên OOD setting
        - TOUR paper (Table 3): TOUR_soft slightly better trên passage retrieval
        - ReSCORE-TTA: mục tiêu OOD → TOUR_hard là lựa chọn hợp lý để thử

    Lưu ý về Dual Pseudo-label với TOUR_hard:
        C_hard được chọn từ Q_TTA (dual pseudo-label) thay vì từ CE thuần.
        Nucleus selection dùng Q_TTA scores làm probability để sort và tích lũy.
        Điều này khác TOUR gốc (dùng CE scores thuần), nhưng consistent với
        framework ReSCORE-TTA.
    """

    def _optimize_query_vector_l1(
        self,
        query_str: str,
        q0: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        TOUR_hard Level 1: Optimize query vector q_t bằng Maximum Marginal
        Likelihood loss trên tập pseudo-positive C_hard.

        Loss (TOUR Eq. 9):
            L_hard(q_t, C^{q_t}_{1:k}) = -log Σ_{c̃ ∈ C_hard} P_k(c̃|q_t)

        Với:
            P_k(c̃|q_t) = exp(sim(q_t, c̃)) / Σ_j exp(sim(q_t, c_j))
            C_hard = nucleus-selected subset of top-M docs theo Q_TTA

        Update rule (Eq. 10):
            q_{t+1} ← q_t - η * ∂L_hard/∂q_t

        Tương đương Rocchio generalized (Eq. 11):
            q_{t+1} = q_t + η * Σ_{c̃} P(c̃|q_t)(1 - P_k(c̃|q_t)) * c̃
                     - η * Σ_{c̃} [P(c̃|q_t) * Σ_{c≠c̃} P_k(c|q_t) * c]

        Early stopping (TOUR §3.5):
            Dừng khi c1 ∈ C^{q_t}_{hard}
            (top-1 của retriever đã nằm trong pseudo-positive set)

        So sánh với _optimize_query_vector_l1() trong TTARetrievalStep (soft):
            Soft: loss = KL(Q_TTA || P_k)  → dùng TOÀN BỘ Q_TTA làm soft target
            Hard: loss = -log Σ P_k(C_hard) → chỉ maximize P_k trên C_hard

        Args:
            query_str: câu truy vấn dạng string.
            q0: initial query vector [d], detached.

        Returns:
            (q_T, stats): optimized vector và dict thống kê.
        """
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
            'variant': 'hard',
            'steps': 0,
            'early_stopped': False,
            'pseudo_label_failures': 0,
            'candidate_refreshes': 0,
        }
        fixed_candidates = None

        for step in range(self.cfg.tta_inner_steps):
            # Retrieve candidates (fix sau lần đầu trừ khi cfg yêu cầu refresh)
            if (
                fixed_candidates is None
                or self.cfg.tta_refresh_candidates_each_step
            ):
                fixed_candidates = self._retrieve_candidates(q_t.detach())
                stats['candidate_refreshes'] += 1
            documents, doc_embeddings = fixed_candidates

            # Tính Q_TTA — giống TOUR_soft, chỉ cách dùng khác
            pseudo_gt = self._compute_pseudo_label(
                query_str=query_str,
                doc_texts=[_doc_to_text(doc) for doc in documents],
                doc_embeddings=doc_embeddings,
                q_current=q_t.detach(),
            )
            if pseudo_gt is None:
                stats['pseudo_label_failures'] += 1
                break

            # ── Chọn C_hard bằng nucleus selection trên Q_TTA ──────
            # C_hard = tập nhỏ nhất s.t. Σ_{c̃ ∈ S} Q_TTA(c̃) >= p
            # Giống TOUR §3.2 Eq. (8) nhưng dùng Q_TTA thay CE scores thuần
            positive_indices = self._nucleus_positive_indices(
                pseudo_gt,
                self.cfg.tta_nucleus_p,
            )

            # ── Early stopping (TOUR §3.5, TOUR_hard variant) ───────
            # Dừng khi top-1 của retriever ∈ C_hard
            # (retriever đã tự nhiên đặt pseudo-positive lên vị trí 1)
            if 0 in positive_indices:
                stats['early_stopped'] = True
                stats['early_stop_inner_step'] = step
                break

            if not positive_indices:
                # Không có pseudo-positive nào — không có gradient signal
                stats['pseudo_label_failures'] += 1
                break

            optimizer.zero_grad()

            # ── TOUR_hard loss (Eq. 9) ───────────────────────────────
            # L_hard = -log Σ_{c̃ ∈ C_hard} P_k(c̃|q_t)
            #
            # P_k(c̃|q_t) = softmax(q_t @ D^T / tau)[c̃]
            # Lấy tổng xác suất của các pseudo-positive documents
            logits = q_t @ doc_embeddings.T  # [M]
            p_retriever = F.softmax(
                logits / self.cfg.tta_temperature, dim=0
            )  # [M]

            # Tạo binary mask cho C_hard
            hard_mask = torch.zeros(
                len(documents), dtype=torch.bool, device=logits.device
            )
            for idx in positive_indices:
                hard_mask[idx] = True

            # Tổng xác suất của pseudo-positive documents
            # Clamp để tránh log(0)
            sum_pseudo_pos_prob = p_retriever[hard_mask].sum().clamp(min=1e-9)

            # L_hard = -log Σ P_k(c̃ ∈ C_hard)
            loss_hard = -torch.log(sum_pseudo_pos_prob)

            # ── Anchor regularization (giữ nguyên như TOUR_soft) ────
            # beta * ||q_t - q_0||^2
            loss_anchor = self.cfg.tta_anchor_weight * (
                q_t - q0.detach()
            ).pow(2).sum()

            loss = loss_hard + loss_anchor
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
                'final_hard_loss': float(loss_hard.detach().item()),
                'final_anchor_loss': float(loss_anchor.detach().item()),
                'learning_rate': float(scheduler.get_last_lr()[0]),
                'nucleus_positive_count': len(positive_indices),
                'sum_pseudo_pos_prob': float(sum_pseudo_pos_prob.detach().item()),
            })

        return q_t.detach(), stats
