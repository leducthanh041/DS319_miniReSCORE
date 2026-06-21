# Ablation and Visualization Utilities

This folder contains offline analysis scripts for comparing baseline inference and TTA inference outputs. These scripts do not run model inference; they only read generated prediction artifacts.

## What it generates

`analyze_tta_outputs.py` produces:

- `report.md`: compact report for paper/reviewer discussion
- `qa_metrics.csv`: EM/F1/answer precision/answer recall
- `mhr_by_hop.csv` and `mhr_by_hop.svg`: MHR@8 trajectory by retrieval hop
- `win_tie_loss.csv`: per-question win/tie/loss for QA and retrieval
- `query_shift_distribution.svg`: distribution of TTA query shifts
- `tta_loss_trend_by_iteration.csv`: L1/L2 loss trend by hop
- `evidence_coverage.csv`: number of gold evidence documents found per question
- `case_studies.md`: selected qualitative examples
- `tta_diagnostic_summary.json`: pseudo-label and optimization sanity checks

## MuSiQue InD example

```bash
BASE=predictions/musique/infer_musique_fair_baseline___llama_3.1_8b_instruct___best_validation/multi_retrieval___inference/prompt_set__1/best
TTA=predictions/musique/infer_tta_musique_both_dual_full___llama_3.1_8b_instruct___best_validation/multi_retrieval___inference/prompt_set__1/best

/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/ReSCORE/bin/python \
  ablation/analyze_tta_outputs.py \
  --baseline_dir "$BASE" \
  --tta_dir "$TTA" \
  --output_dir ablation/results/musique_ind \
  --case_count 8 \
  --top_docs 8
```

Open the main report:

```bash
less ablation/results/musique_ind/report.md
```

## OOD example

Use the corresponding baseline/TTA output directories for HotpotQA or 2WikiMultiHopQA:

```bash
/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/ReSCORE/bin/python \
  ablation/analyze_tta_outputs.py \
  --baseline_dir predictions/hotpotqa/<baseline_run>/multi_retrieval___inference/prompt_set__1/best \
  --tta_dir predictions/hotpotqa/<tta_run>/multi_retrieval___inference/prompt_set__1/best \
  --output_dir ablation/results/hotpotqa_ood
```

## L1/L2 ablation summary

After running separate TTA variants, pass their output directories:

```bash
/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/ReSCORE/bin/python \
  ablation/analyze_tta_outputs.py \
  --baseline_dir "$BASE" \
  --tta_dir "$TTA" \
  --output_dir ablation/results/musique_ablation \
  --ablation_run baseline="$BASE" \
  --ablation_run l1_only=predictions/musique/<l1_run>/multi_retrieval___inference/prompt_set__1/best \
  --ablation_run l2_only=predictions/musique/<l2_run>/multi_retrieval___inference/prompt_set__1/best \
  --ablation_run l1_l2="$TTA"
```

Suggested TTA settings for ablations:

- L1-only: `--tta_level l1`
- L2-only: `--tta_level l2`
- L1+L2: `--tta_level both`

Keep all other inference settings identical.

## When to rerun inference

You do not need to rerun the current MuSiQue InD baseline/TTA to generate plots and case studies because both output directories already contain prediction, retrieval trace, retrieval evaluation, and TTA diagnostics.

You should rerun only when:

- The baseline and TTA `count` differ.
- Baseline was not run with the same retrieval/generation prompt budget as TTA.
- You want L1-only and L2-only ablation results.
- You need OOD results for HotpotQA or 2WikiMultiHopQA.

