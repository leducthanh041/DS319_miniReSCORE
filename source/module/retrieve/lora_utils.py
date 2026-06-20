"""
lora_utils.py
=============
LoRA (Low-Rank Adaptation) injection và management cho query encoder
trong ReSCORE-TTA Level 2.

Design decisions:
  - LoRALinear thay thế nn.Linear trong-place, giữ W0 frozen.
  - Khởi tạo: B=0 → delta_W = B@A = 0 → behavior y hệt pretrained lúc đầu.
  - Per-instance reset: gọi reset_lora() trước mỗi test instance mới.
  - Chỉ inject vào N top transformer layers (giảm compute overhead).
  - Target modules mặc định: 'query', 'value' (attention Q,V projections)
    theo LoRA paper (Hu et al., 2022).

References:
  - Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
  - ReSCORE-TTA framework: Level 2 design (lora_utils.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Iterator, List, Optional


# ──────────────────────────────────────────────────────────────────
# Core LoRA Module
# ──────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement cho nn.Linear với LoRA adaptation.

    Forward computation:
        y = x @ W0^T + (x @ A^T) @ B^T * scaling
          = W0(x)  +  B(A(x)) * (alpha / rank)

    Parameter counts:
        W0: [out, in]  — frozen (pretrained weights, shared reference)
        A:  [rank, in] — trainable LoRA matrix
        B:  [out, rank]— trainable LoRA matrix, initialized to 0

    Tại init: B=0 → delta_W = B@A = 0 → y = W0(x) (không thay đổi output)
    Sau training: B ≠ 0 → y = W0(x) + B(A(x)) * scaling (adapted output)
    Sau reset: B=0 lại → y = W0(x) (back to pretrained behavior)

    Args:
        original_linear: nn.Linear gốc cần được replace.
        rank: LoRA rank r. Điển hình: 4, 8, 16.
        lora_alpha: scaling factor. scaling = lora_alpha / rank.
                    LoRA paper thường dùng alpha = 2 * rank.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Giữ W0 và bias — chia sẻ tensor với linear gốc, freeze
        self.weight = original_linear.weight      # [out, in], FROZEN
        self.bias = original_linear.bias          # [out] or None, FROZEN
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = lora_alpha / rank
        self.enabled = True

        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA matrices — HAI PARAMETER NÀY được update trong TTA
        # A: khởi tạo Kaiming uniform (như LoRA paper §4)
        # B: khởi tạo zeros (đảm bảo delta_W=0 tại t=0)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.register_buffer(
            "_initial_lora_A",
            self.lora_A.detach().clone(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear (frozen W0)
        base_out = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return base_out
        # LoRA branch: (x @ A^T) @ B^T * scaling
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def reset(self):
        """Restore initial LoRA parameters for per-instance isolation."""
        with torch.no_grad():
            self.lora_A.copy_(self._initial_lora_A)
            self.lora_B.zero_()

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


# ──────────────────────────────────────────────────────────────────
# LoRA Injection
# ──────────────────────────────────────────────────────────────────

def inject_lora(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    num_top_layers: int = 4,
) -> nn.Module:
    """
    Inject LoRA vào query encoder (in-place).

    Chỉ inject vào `num_top_layers` layers cuối cùng của transformer
    để giảm overhead. Trong TTA, top layers thường mang information
    quan trọng nhất về query semantics.

    Args:
        model: query_model của DenseRetriever (AutoModel / BertModel).
        rank: LoRA rank r. Khuyến nghị: 8.
        lora_alpha: LoRA scaling alpha. Khuyến nghị: 16.0.
        target_modules: tên các submodule để inject.
            Default = ['query', 'value'] (Q, V projections của self-attention).
            Theo LoRA paper: chỉ Q, V là đủ; thêm K, O không cải thiện nhiều.
        num_top_layers: số layers cuối cùng để inject.
            4 là default hợp lý cho Contriever (12 layers).

    Returns:
        model với LoRA injected (in-place, cũng return để tiện chaining).

    Ví dụ:
        retriever.query_model = inject_lora(
            retriever.query_model, rank=8, num_top_layers=4
        )
    """
    if target_modules is None:
        target_modules = ['query', 'value']

    # Tìm encoder layers
    # Contriever = facebook/contriever → BertModel structure
    encoder_layers = _find_encoder_layers(model)

    if encoder_layers is not None:
        total = len(encoder_layers)
        start = max(0, total - num_top_layers)
        injected_count = 0
        for layer_idx in range(start, total):
            layer = encoder_layers[layer_idx]
            count = _inject_into_layer(layer, rank, lora_alpha, target_modules)
            injected_count += count
        print(
            f"[inject_lora] Injected {injected_count} LoRALinear modules "
            f"into layers [{start}..{total - 1}] "
            f"(top {num_top_layers} of {total}), "
            f"rank={rank}, alpha={lora_alpha}"
        )
    else:
        # Fallback: scan toàn bộ model và inject vào Linear phù hợp
        injected_count = _inject_fallback(model, rank, lora_alpha, target_modules)
        print(
            f"[inject_lora] Fallback injection: {injected_count} LoRALinear modules, "
            f"rank={rank}, alpha={lora_alpha}"
        )

    if injected_count == 0:
        print(
            f"[inject_lora] WARNING: 0 modules injected. "
            f"Check target_modules={target_modules} and model architecture."
        )

    return model


def _find_encoder_layers(model: nn.Module):
    """Tìm list các transformer encoder layers."""
    # BertModel (Contriever): model.encoder.layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    # RoBERTa: model.roberta.encoder.layer
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return model.roberta.encoder.layer
    # Generic: model.layers
    if hasattr(model, 'layers'):
        return model.layers
    return None


def _inject_into_layer(
    layer: nn.Module,
    rank: int,
    lora_alpha: float,
    target_modules: List[str],
) -> int:
    """Inject LoRA vào một transformer layer. Returns số modules đã inject."""
    count = 0
    # BertLayer: layer.attention.self.query / layer.attention.self.value
    if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
        self_attn = layer.attention.self
        for module_name in target_modules:
            if hasattr(self_attn, module_name):
                original = getattr(self_attn, module_name)
                if isinstance(original, nn.Linear):
                    setattr(
                        self_attn,
                        module_name,
                        LoRALinear(original, rank=rank, lora_alpha=lora_alpha),
                    )
                    count += 1
    return count


def _inject_fallback(
    model: nn.Module,
    rank: int,
    lora_alpha: float,
    target_modules: List[str],
) -> int:
    """Fallback injection: scan toàn bộ named modules."""
    count = 0
    for full_name, module in list(model.named_modules()):
        for target in target_modules:
            if target in full_name.split('.')[-1] and isinstance(module, nn.Linear):
                # Tìm parent module
                parts = full_name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    try:
                        parent = model.get_submodule(parent_name)
                        setattr(
                            parent,
                            child_name,
                            LoRALinear(module, rank=rank, lora_alpha=lora_alpha),
                        )
                        count += 1
                    except Exception:
                        pass
    return count


# ──────────────────────────────────────────────────────────────────
# LoRA State Management
# ──────────────────────────────────────────────────────────────────

def reset_lora(model: nn.Module):
    """
    Reset tất cả LoRA B matrices về 0.

    Phải gọi tại đầu mỗi test instance mới để đảm bảo
    per-instance isolation (không có gradient "lây" giữa instances).

    Sau khi gọi: model behavior = pretrained encoder (delta_W = 0).
    """
    reset_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset()
            reset_count += 1
    return reset_count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Trả về list tất cả LoRA parameters (lora_A và lora_B).

    Dùng để tạo optimizer chỉ update LoRA, không đụng vào W0:
        optimizer = Adam(get_lora_parameters(model), lr=5e-4)

    Returns:
        List[nn.Parameter] chứa tất cả lora_A và lora_B tensors.
        Empty list nếu không có LoRA nào được inject.
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def mark_only_lora_as_trainable(model: nn.Module) -> int:
    """Freeze the base encoder and leave only LoRA A/B trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            trainable_count += module.lora_A.numel() + module.lora_B.numel()
    return trainable_count


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily disable every LoRA branch without modifying parameters."""
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    previous_states = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous_states):
            module.enabled = enabled


def lora_norm_regularization(model: nn.Module) -> torch.Tensor:
    """
    Tính ||B @ A||_F^2 tổng cộng trên tất cả LoRA layers.

    Dùng làm L_LoRA-reg trong loss Eq. (22):
        L_LoRA-reg = gamma * sum_layers ||B_l @ A_l||_F^2

    Giới hạn magnitude của LoRA update, tránh parameter drift quá lớn.

    Returns:
        Scalar tensor (requires_grad=True nếu LoRA params có grad).
    """
    device = next(
        (p for m in model.modules() if isinstance(m, LoRALinear) for p in [m.lora_B]),
        torch.tensor(0.0),
    )
    total_norm = torch.zeros(1, device=device.device if isinstance(device, torch.Tensor) else 'cpu')

    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta_w = module.lora_B @ module.lora_A   # [out, in]
            total_norm = total_norm + delta_w.pow(2).sum()

    return total_norm.squeeze()


def count_lora_parameters(model: nn.Module) -> dict:
    """
    Đếm số parameters LoRA và tổng parameters.
    Hữu ích để verify injection đúng.

    Returns:
        dict với 'lora', 'frozen', 'total' counts.
    """
    lora_params = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in [m.lora_A, m.lora_B]
    )
    frozen_params = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'lora': lora_params,
        'frozen': frozen_params,
        'total': total_params,
        'lora_pct': 100.0 * lora_params / max(total_params, 1),
    }
