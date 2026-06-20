"""
test_tta_components.py
======================
Script test nhanh từng component của ReSCORE-TTA.
Chạy từ root của project DS319_miniReSCORE-main:

    python test_tta_components.py

Sẽ test lần lượt:
    [1] lora_utils: inject, reset, get_parameters
    [2] cross_encoder_wrapper: score_documents, soft_labels
    [3] dense_patch: embed_single_query_no_detach (nếu đã patch dense.py)
    [4] config_patch: TTA fields trong PipelineConfig (nếu đã patch config.py)
    [5] tta_retrieval: import check (không cần full pipeline)

Mỗi test in [OK] hoặc [FAIL] với mô tả.
"""

import sys
import traceback
import torch


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  [OK]   {msg}")


def fail(msg: str, exc: Exception = None):
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
# [1] lora_utils
# ══════════════════════════════════════════════════════════════════
section("[1] lora_utils")

try:
    from source.module.retrieve.lora_utils import (
        LoRALinear,
        inject_lora,
        reset_lora,
        get_lora_parameters,
        lora_norm_regularization,
        count_lora_parameters,
    )
    ok("import lora_utils OK")
except Exception as e:
    fail("import lora_utils", e)
    sys.exit(1)

try:
    from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig

    cfg = DenseRetrieverConfig(
        query_model_name_or_path='facebook/contriever-msmarco',
        training_strategy=None,
        use_fp16=False,
        device='cpu',
    )
    retriever = DenseRetriever(cfg)
    ok("DenseRetriever loaded on CPU")
except Exception as e:
    fail("DenseRetriever init", e)
    sys.exit(1)

try:
    retriever.query_model = inject_lora(
        retriever.query_model, rank=4, num_top_layers=2
    )
    ok("inject_lora completed")
except Exception as e:
    fail("inject_lora", e)

try:
    lora_params = get_lora_parameters(retriever.query_model)
    assert len(lora_params) > 0, "No LoRA parameters found!"
    total_lora = sum(p.numel() for p in lora_params)
    ok(f"get_lora_parameters: {len(lora_params)} tensors, {total_lora:,} params")
except Exception as e:
    fail("get_lora_parameters", e)

try:
    stats = count_lora_parameters(retriever.query_model)
    ok(
        f"count_lora_parameters: "
        f"lora={stats['lora']:,}, total={stats['total']:,}, "
        f"pct={stats['lora_pct']:.3f}%"
    )
except Exception as e:
    fail("count_lora_parameters", e)

try:
    n = reset_lora(retriever.query_model)
    ok(f"reset_lora: reset {n} LoRALinear modules")
    # Verify B matrices are zero
    from source.module.retrieve.lora_utils import LoRALinear
    for m in retriever.query_model.modules():
        if isinstance(m, LoRALinear):
            assert m.lora_B.abs().max().item() < 1e-9, "B not zero after reset!"
    ok("Verify B=0 after reset: PASSED")
except Exception as e:
    fail("reset_lora", e)

try:
    lora_params = get_lora_parameters(retriever.query_model)
    optimizer = torch.optim.Adam(lora_params, lr=5e-4)
    norm = lora_norm_regularization(retriever.query_model)
    # norm should be 0 since B=0 after reset
    ok(f"lora_norm_regularization after reset: {norm.item():.6f} (expect ~0.0)")
except Exception as e:
    fail("lora_norm_regularization", e)


# ══════════════════════════════════════════════════════════════════
# [2] cross_encoder_wrapper
# ══════════════════════════════════════════════════════════════════
section("[2] cross_encoder_wrapper")

try:
    from source.module.retrieve.cross_encoder_wrapper import CrossEncoderWrapper
    ok("import CrossEncoderWrapper OK")
except Exception as e:
    fail("import CrossEncoderWrapper", e)
    sys.exit(1)

try:
    ce = CrossEncoderWrapper(
        model_name_or_path='cross-encoder/ms-marco-MiniLM-L-6-v2',
        device='cpu',
        batch_size=8,
    )
    ok("CrossEncoderWrapper init OK")
except Exception as e:
    fail("CrossEncoderWrapper init (make sure cross-encoder model is available)", e)
    print("  → Install: pip install sentence-transformers")
    ce = None

if ce is not None:
    try:
        query = "What is the capital of France?"
        docs = [
            "Paris is the capital and largest city of France.",
            "The weather today is sunny and warm.",
            "France is a country in Western Europe.",
        ]
        scores = ce.score_documents(query, docs)
        assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"
        ok(f"score_documents: {scores.tolist()}")
        # Paris document should score highest
        assert scores.argmax().item() == 0, "Expected doc 0 (Paris) to score highest!"
        ok("Relevance ranking sanity check: PASSED (Paris doc is top-1)")
    except Exception as e:
        fail("score_documents", e)

    try:
        soft = ce.get_soft_labels(query, docs, tau=0.5)
        assert abs(soft.sum().item() - 1.0) < 1e-5, f"Soft labels do not sum to 1: {soft.sum()}"
        ok(f"get_soft_labels: {soft.tolist()} (sum={soft.sum():.4f})")
    except Exception as e:
        fail("get_soft_labels", e)

    try:
        hard = ce.get_hard_labels(query, docs, tau=0.5, p=0.5)
        assert isinstance(hard, list), "get_hard_labels should return list"
        assert len(hard) >= 1, "Hard labels should have at least 1 element"
        ok(f"get_hard_labels: indices={hard}")
    except Exception as e:
        fail("get_hard_labels", e)

    try:
        # Test cache: calling again should use cache
        scores2 = ce.score_documents(query, docs)
        assert torch.allclose(scores, scores2), "Cache inconsistency!"
        ok(f"Cache test: consistent results, cache_size={len(ce._cache)}")
        ce.clear_cache()
        ok("clear_cache: OK")
    except Exception as e:
        fail("Cache test", e)


# ══════════════════════════════════════════════════════════════════
# [3] dense_patch — embed_single_query_no_detach
# ══════════════════════════════════════════════════════════════════
section("[3] dense_patch (embed_single_query_no_detach)")

try:
    # Test if method was added to DenseRetriever
    if hasattr(retriever, 'embed_single_query_no_detach'):
        ok("embed_single_query_no_detach method found")

        # Test output shape
        retriever.query_model.eval()
        with torch.no_grad():
            emb = retriever.embed_single_query_no_detach(
                "What is multi-hop question answering?"
            )
        assert emb.ndim == 1, f"Expected 1D tensor, got shape {emb.shape}"
        ok(f"Output shape: {emb.shape}  (expected: [768] for Contriever)")

        # Test Level 1 usage pattern
        q0 = emb.detach().clone().requires_grad_(True)
        assert q0.requires_grad, "q0 should require grad"
        ok("Level 1 pattern: detach().clone().requires_grad_(True) → OK")

        # Test Level 2 usage pattern (gradient flows through model)
        retriever.query_model.train()
        lora_params = get_lora_parameters(retriever.query_model)
        if lora_params:
            optimizer_test = torch.optim.Adam(lora_params, lr=1e-4)
            optimizer_test.zero_grad()
            emb_grad = retriever.embed_single_query_no_detach(
                "What is multi-hop question answering?"
            )
            loss = emb_grad.sum()
            loss.backward()
            grad_found = any(p.grad is not None for p in lora_params)
            ok(f"Level 2 pattern: gradient flows to LoRA params: {grad_found}")
        retriever.query_model.eval()
    else:
        fail(
            "embed_single_query_no_detach NOT FOUND in DenseRetriever. "
            "Please apply dense_patch.py to source/module/retrieve/dense.py"
        )
except Exception as e:
    fail("dense_patch test", e)


# ══════════════════════════════════════════════════════════════════
# [4] config_patch — TTA fields
# ══════════════════════════════════════════════════════════════════
section("[4] config_patch (PipelineConfig TTA fields)")

try:
    from source.pipeline.config import PipelineConfig

    # Test if TTA fields exist
    cfg_test = PipelineConfig()
    tta_fields = [
        'use_tta', 'tta_level', 'tta_pseudo_label',
        'tta_cross_encoder_model', 'tta_inner_steps',
        'tta_query_lr', 'tta_lora_rank', 'tta_lora_lr',
    ]
    missing = [f for f in tta_fields if not hasattr(cfg_test, f)]
    if missing:
        fail(
            f"Missing TTA fields in PipelineConfig: {missing}. "
            "Please apply config_patch.py to source/pipeline/config.py"
        )
    else:
        ok(f"All TTA fields found in PipelineConfig: {tta_fields}")

    # Test properties
    if hasattr(cfg_test, 'tta_q_rel_input_prompt_file_path'):
        path = cfg_test.tta_q_rel_input_prompt_file_path
        ok(f"tta_q_rel_input_prompt_file_path: {path}")
    else:
        fail("tta_q_rel_input_prompt_file_path property missing")

except Exception as e:
    fail("config_patch test", e)


# ══════════════════════════════════════════════════════════════════
# [5] tta_retrieval import check
# ══════════════════════════════════════════════════════════════════
section("[5] tta_retrieval import check")

try:
    from source.pipeline.step.tta_retrieval import TTARetrievalStep, _doc_to_text
    ok("import TTARetrievalStep OK")

    # Test _doc_to_text utility
    from source.module.index.docstore import Document
    doc = Document(
        id="test_1",
        content="Sample content",
        metadata={"title": "Sample Title", "text": "Sample text body"},
    )
    text = _doc_to_text(doc)
    assert "Sample Title" in text
    ok(f"_doc_to_text: '{text[:60]}'")

except Exception as e:
    fail("tta_retrieval import", e)


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
section("SUMMARY")
print("""
All tests completed. If all components show [OK]:
    → Run a quick end-to-end test with --demo flag:

    python source/run/inference_tta.py \\
        --method iqatr_tta_demo \\
        --dataset musique \\
        --dataset_split test \\
        --demo \\
        --retrieval_query_model_name_or_path facebook/contriever-msmarco \\
        --generation_backend vllm_server \\
        --tta_level l1 \\
        --tta_pseudo_label ce_only \\
        --tta_inner_steps 1
""")
