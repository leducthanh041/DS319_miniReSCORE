# Baseline vs TTA Analysis Report

- Baseline: `predictions/musique/infer_musique_fair_baseline___llama_3.1_8b_instruct___best_validation/multi_retrieval___inference/prompt_set__1/best`
- TTA: `predictions/musique/infer_tta_musique_both_dual_full___llama_3.1_8b_instruct___best_validation/multi_retrieval___inference/prompt_set__1/best`

## QA metrics

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 9.60 | 10.20 | +0.60 |
| f1 | 17.70 | 20.30 | +2.60 |
| precision | 19.20 | 21.20 | +2.00 |
| recall | 20.90 | 25.90 | +5.00 |

## MHR by hop

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| MHR_1@8 | 25.22 | 30.53 | +5.32 |
| MHR_2@8 | 26.17 | 34.17 | +8.00 |
| MHR_3@8 | 26.22 | 34.63 | +8.42 |
| MHR_4@8 | 26.22 | 34.80 | +8.58 |
| MHR_5@8 | 26.22 | 34.90 | +8.68 |
| MHR_6@8 | 26.22 | 34.90 | +8.68 |
| MHR_7@8 | 26.22 | 34.90 | +8.68 |
| MHR_final@8 | 26.22 | 34.90 | +8.68 |
| title_only_MHR_1@8 | 29.57 | 33.83 | +4.27 |
| title_only_MHR_2@8 | 31.23 | 38.48 | +7.25 |
| title_only_MHR_3@8 | 31.38 | 39.02 | +7.63 |
| title_only_MHR_4@8 | 31.38 | 39.12 | +7.73 |
| title_only_MHR_5@8 | 31.38 | 39.22 | +7.83 |
| title_only_MHR_6@8 | 31.38 | 39.22 | +7.83 |
| title_only_MHR_7@8 | 31.38 | 39.33 | +7.95 |
| title_only_MHR_final@8 | 31.38 | 39.33 | +7.95 |

## Win / Tie / Loss

| Metric | Win | Tie | Loss | Mean delta |
|---|---:|---:|---:|---:|
| answer_em | 19 | 465 | 16 | 0.0060 |
| answer_f1 | 95 | 353 | 52 | 0.0257 |
| answer_precision | 90 | 358 | 52 | 0.0199 |
| answer_recall | 74 | 390 | 36 | 0.0504 |
| MHR_final | 104 | 375 | 21 | 0.0868 |
| title_only_MHR_final | 100 | 372 | 28 | 0.0795 |

## TTA diagnostics

- `trace_records`: 1974
- `unique_questions`: 500
- `pseudo_ok_rate`: 1.0
- `query_shift_l2`: count=1974, mean=1.4177, median=1.2086, min=0.0086, max=3.1779, p10=0.0200, p90=2.4543
- `l1_loss`: count=1671, mean=2.7571, median=2.8811, min=0.3190, max=4.9757, p10=1.6826, p90=3.4864
- `l2_loss`: count=1974, mean=2.6021, median=2.8228, min=0.2465, max=3.9596, p10=1.4625, p90=3.3031
- `mean_l1_steps`: 1.5374873353596759

## Ablation runs

No L1-only/L2-only ablation runs were provided. Run them separately and pass `--ablation_run name=path`.

## Generated files

- `qa_metrics`: `ablation/results/musique_ind/qa_metrics.csv`
- `mhr_by_hop`: `ablation/results/musique_ind/mhr_by_hop.csv`
- `mhr_by_hop_plot`: `ablation/results/musique_ind/mhr_by_hop.svg`
- `win_tie_loss`: `ablation/results/musique_ind/win_tie_loss.csv`
- `query_shift_distribution`: `ablation/results/musique_ind/query_shift_distribution.svg`
- `tta_loss_trend`: `ablation/results/musique_ind/tta_loss_trend_by_iteration.csv`
- `evidence_coverage`: `ablation/results/musique_ind/evidence_coverage.csv`
- `case_studies`: `ablation/results/musique_ind/case_studies.md`
- `tta_diagnostic_summary`: `ablation/results/musique_ind/tta_diagnostic_summary.json`