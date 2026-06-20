"""
config_patch.py
===============
Patch này chứa các FIELDS VÀ PROPERTIES mới cần thêm vào
class PipelineConfig trong source/pipeline/config.py.

HƯỚNG DẪN ÁP DỤNG:

BƯỚC 1: Mở source/pipeline/config.py, tìm @dataclass PipelineConfig.
        Tìm dòng khai báo cuối cùng của các fields hiện có
        (ví dụ: `wandb_key`, `n_epochs`, v.v.).
        Thêm các TTA fields bên dưới.

BƯỚC 2: Tìm các @property hiện có trong PipelineConfig
        (ví dụ: `qa_gen_input_prompt_file_path`).
        Thêm hai properties mới bên cạnh.

KHÔNG thay đổi bất kỳ field hay property hiện có.
"""

# ══════════════════════════════════════════════════════════════════
# BƯỚC 1: THÊM VÀO PHẦN DATACLASS FIELDS (cuối @dataclass PipelineConfig)
# ══════════════════════════════════════════════════════════════════
#
# Tìm dòng cuối cùng trong phần field declarations, paste sau đó:

TTA_FIELDS = """
    # ── TTA (Test-Time Adaptation) ────────────────────────────────
    # General
    use_tta: Optional[bool] = False
    tta_level: Optional[str] = 'both'
    # Choices: 'l1' (query vector only), 'l2' (LoRA only), 'both'

    tta_pseudo_label: Optional[str] = 'dual'
    # Choices: 'ce_only' (cross-encoder only, fastest),
    #          'lm_only' (P_LM(q|d) only),
    #          'dual'   (CE * LM_rel, best quality, slowest)

    # Cross-encoder (dùng cho CE signal trong pseudo-label)
    tta_cross_encoder_model: Optional[str] = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    tta_cross_encoder_batch_size: Optional[int] = 32

    # Level 1: Query Vector Optimization
    # Hyperparameters từ TOUR paper (Table 7, Appendix D):
    tta_inner_steps: Optional[int] = 3          # T_inner — max gradient steps/hop
    tta_query_lr: Optional[float] = 1.2         # eta_q — TOUR: 1.2 for DensePhrases
    tta_momentum: Optional[float] = 0.99        # SGD momentum — TOUR Appendix D
    tta_weight_decay: Optional[float] = 0.01    # weight decay — TOUR Appendix D
    tta_temperature: Optional[float] = 0.5      # tau — CE softmax temperature
    tta_nucleus_p: Optional[float] = 0.5        # p — nucleus threshold (hard labels)
    tta_anchor_weight: Optional[float] = 0.1    # beta — anchor regularization

    # Level 2: LoRA Adaptation
    tta_lora_rank: Optional[int] = 8            # r — LoRA rank
    tta_lora_alpha: Optional[float] = 16.0      # lora_alpha — scaling = alpha/rank
    tta_lora_lr: Optional[float] = 5e-4         # eta_LoRA — Adam learning rate
    tta_lora_num_top_layers: Optional[int] = 4  # N top transformer layers để inject
    tta_lora_reg_weight: Optional[float] = 0.01  # gamma — LoRA norm regularization
"""

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2: THÊM VÀO PHẦN @property (bên cạnh các property hiện có)
# ══════════════════════════════════════════════════════════════════
#
# Tìm phần @property trong PipelineConfig (ví dụ: qa_gen_input_prompt_file_path),
# thêm hai properties sau vào cùng chỗ:

TTA_PROPERTIES = """
    @property
    def tta_q_rel_input_prompt_file_path(self) -> str:
        \"\"\"
        Path đến prompt condition cho P_LM(q|d).
        File này yêu cầu LLM sinh câu hỏi từ document.
        (Xem prompts/prompt_set__1/q_rel_input.txt)
        \"\"\"
        return f'./prompts/prompt_set__{self.prompt_set}/q_rel_input.txt'

    @property
    def tta_q_rel_output_prompt_file_path(self) -> str:
        \"\"\"
        Path đến prediction template cho P_LM(q|d).
        Chứa format: {\"question\": \"{question}\"} để tính perplexity.
        (Xem prompts/prompt_set__1/q_rel_output.txt)
        \"\"\"
        return f'./prompts/prompt_set__{self.prompt_set}/q_rel_output.txt'
"""

# ══════════════════════════════════════════════════════════════════
# VERIFY sau khi patch:
# ══════════════════════════════════════════════════════════════════
#
# from source.pipeline.config import PipelineConfig
#
# cfg = PipelineConfig(use_tta=True, tta_level='both', tta_lora_rank=8)
# assert cfg.use_tta == True
# assert cfg.tta_lora_rank == 8
# print(f"[OK] tta_q_rel_input_prompt_file_path: {cfg.tta_q_rel_input_prompt_file_path}")
# # Expected: ./prompts/prompt_set__1/q_rel_input.txt
