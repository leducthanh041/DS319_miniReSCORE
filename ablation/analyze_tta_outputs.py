#!/usr/bin/env python3
"""Offline analysis for baseline vs TTA inference outputs.

The script reads existing prediction directories and produces tables, plots,
and case-study markdown files without running model inference.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.evaluation.evaluate import evaluate_by_dicts


MetricDict = Dict[str, float]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.lower() in {"n/a", "nan", ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def delta(a: float, b: float) -> str:
    return f"{100.0 * (b - a):+.2f}"


def normalize_qid(value: Any) -> str:
    return str(value)


def load_trace_by_qid(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(path):
        qid = normalize_qid(record.get("question_id"))
        record["iteration"] = int(record.get("iteration", 0))
        grouped[qid].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.get("iteration", 0))
    return dict(grouped)


def load_per_question(path: Path) -> Dict[str, Dict[str, Any]]:
    return {normalize_qid(row["question_id"]): row for row in iter_jsonl(path)}


def infer_dataset_from_config(run_dir: Path) -> Optional[str]:
    config_path = run_dir / "configuration.json"
    if not config_path.exists():
        return None
    return read_json(config_path).get("dataset")


def load_processed_data(dataset: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not dataset:
        return {}
    candidates = [
        Path("data/processed_data") / dataset / "test_subsampled.jsonl",
        Path("data/processed_data") / dataset / "test.jsonl",
        Path("data/processed_data") / dataset / "dev_subsampled.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return {
                normalize_qid(item["question_id"]): item
                for item in iter_jsonl(path)
            }
    return {}


def per_question_answer_metrics(
    predictions: Dict[str, Any],
    ground_truths: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    rows = {}
    for qid in sorted(set(predictions) & set(ground_truths)):
        metrics = evaluate_by_dicts(
            prediction_type="answer",
            id_to_ground_truths={qid: ground_truths[qid]},
            id_to_predictions={qid: predictions[qid]},
        )
        rows[qid] = {
            "em": safe_float(metrics.get("em")),
            "f1": safe_float(metrics.get("f1")),
            "precision": safe_float(metrics.get("precision")),
            "recall": safe_float(metrics.get("recall")),
        }
    return rows


def extract_mhr_rows(base_eval: Dict[str, Any], tta_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
    pattern = re.compile(r"^(title_only_)?MHR_(.+)@(\d+)$")
    keys = []
    for key in sorted(set(base_eval) | set(tta_eval)):
        match = pattern.match(key)
        if match:
            title_only = bool(match.group(1))
            idx = match.group(2)
            sort_idx = 10_000 if idx == "final" else int(idx)
            keys.append((title_only, sort_idx, key))
    rows = []
    for title_only, _, key in keys:
        base = safe_float(base_eval.get(key))
        tta = safe_float(tta_eval.get(key))
        rows.append(
            {
                "metric": key,
                "type": "title_only" if title_only else "strict",
                "baseline": base,
                "tta": tta,
                "delta": tta - base,
                "baseline_pct": pct(base),
                "tta_pct": pct(tta),
                "delta_points": delta(base, tta),
            }
        )
    return rows


def build_win_tie_loss(
    base_answer: Dict[str, Dict[str, float]],
    tta_answer: Dict[str, Dict[str, float]],
    base_pq: Dict[str, Dict[str, Any]],
    tta_pq: Dict[str, Dict[str, Any]],
    eps: float,
) -> List[Dict[str, Any]]:
    specs = [
        ("answer_em", lambda q: base_answer[q]["em"], lambda q: tta_answer[q]["em"]),
        ("answer_f1", lambda q: base_answer[q]["f1"], lambda q: tta_answer[q]["f1"]),
        (
            "answer_precision",
            lambda q: base_answer[q]["precision"],
            lambda q: tta_answer[q]["precision"],
        ),
        (
            "answer_recall",
            lambda q: base_answer[q]["recall"],
            lambda q: tta_answer[q]["recall"],
        ),
        (
            "MHR_final",
            lambda q: safe_float(base_pq[q].get("final_recall")),
            lambda q: safe_float(tta_pq[q].get("final_recall")),
        ),
        (
            "title_only_MHR_final",
            lambda q: safe_float(base_pq[q].get("title_only_final_recall")),
            lambda q: safe_float(tta_pq[q].get("title_only_final_recall")),
        ),
    ]
    rows = []
    common_answer = set(base_answer) & set(tta_answer)
    common_retrieval = set(base_pq) & set(tta_pq)
    for metric, base_fn, tta_fn in specs:
        qids = common_retrieval if "MHR" in metric else common_answer
        win = tie = loss = 0
        total_delta = 0.0
        for qid in qids:
            d = tta_fn(qid) - base_fn(qid)
            total_delta += d
            if d > eps:
                win += 1
            elif d < -eps:
                loss += 1
            else:
                tie += 1
        count = len(qids)
        rows.append(
            {
                "metric": metric,
                "win": win,
                "tie": tie,
                "loss": loss,
                "count": count,
                "win_rate": win / count if count else 0.0,
                "loss_rate": loss / count if count else 0.0,
                "mean_delta": total_delta / count if count else 0.0,
            }
        )
    return rows


def evidence_coverage_rows(
    label: str,
    per_question: Dict[str, Dict[str, Any]],
    title_only: bool = False,
) -> List[Dict[str, Any]]:
    counter: Counter[int] = Counter()
    total = 0
    for row in per_question.values():
        supports = int(row.get("num_gold_supports") or 0)
        if supports <= 0:
            continue
        recall_key = "title_only_final_recall" if title_only else "final_recall"
        matched = int(round(safe_float(row.get(recall_key)) * supports))
        counter[matched] += 1
        total += 1
    return [
        {
            "setting": label,
            "matched_gold_evidence": matched,
            "question_count": count,
            "question_rate": count / total if total else 0.0,
            "title_only": title_only,
        }
        for matched, count in sorted(counter.items())
    ]


def collect_tta_diagnostics(trace_by_qid: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    by_iteration: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for qid, records in trace_by_qid.items():
        for record in records:
            adaptation = record.get("adaptation") or {}
            l1 = adaptation.get("l1") or {}
            l2 = adaptation.get("l2") or {}
            query_shift = safe_float(adaptation.get("query_shift_l2"), math.nan)
            l1_loss = safe_float(l1.get("final_loss"), math.nan)
            l2_loss = safe_float(l2.get("loss"), math.nan)
            row = {
                "question_id": qid,
                "iteration": int(record.get("iteration", 0)),
                "pseudo_label_ok": bool(record.get("pseudo_label_ok", False)),
                "query_shift_l2": query_shift,
                "l1_steps": int(l1.get("steps") or 0),
                "l1_loss": l1_loss,
                "l1_early_stopped": bool(l1.get("early_stopped", False)),
                "l2_updated": bool(l2.get("updated", False)),
                "l2_loss": l2_loss,
                "l2_kl_loss": safe_float(l2.get("kl_loss"), math.nan),
                "l2_anchor_loss": safe_float(l2.get("anchor_loss"), math.nan),
                "l2_regularization_loss": safe_float(l2.get("regularization_loss"), math.nan),
            }
            rows.append(row)
            iteration = row["iteration"]
            for key in ["query_shift_l2", "l1_loss", "l2_loss", "l2_kl_loss"]:
                value = row[key]
                if not math.isnan(value):
                    by_iteration[iteration][key].append(value)
    trend_rows = []
    for iteration, values in sorted(by_iteration.items()):
        trend = {"iteration": iteration}
        for key, vals in values.items():
            trend[f"mean_{key}"] = mean(vals) if vals else math.nan
            trend[f"median_{key}"] = median(vals) if vals else math.nan
            trend[f"count_{key}"] = len(vals)
        trend_rows.append(trend)
    return rows, trend_rows


def summary_stats(values: List[float]) -> Dict[str, Any]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"count": 0}
    clean_sorted = sorted(clean)
    def q(p: float) -> float:
        idx = min(len(clean_sorted) - 1, max(0, int(round((len(clean_sorted) - 1) * p))))
        return clean_sorted[idx]
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
        "p10": q(0.10),
        "p90": q(0.90),
    }


def svg_line_chart(path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    strict_rows = [r for r in rows if r["type"] == "strict"]
    if not strict_rows:
        return
    width, height = 760, 420
    margin = 60
    values = [r["baseline"] for r in strict_rows] + [r["tta"] for r in strict_rows]
    ymin, ymax = 0.0, max(values + [0.01]) * 1.15
    labels = [r["metric"].replace("MHR_", "").replace("@8", "") for r in strict_rows]
    n = len(strict_rows)
    def x(i: int) -> float:
        if n == 1:
            return width / 2
        return margin + i * (width - 2 * margin) / (n - 1)
    def y(v: float) -> float:
        return height - margin - (v - ymin) * (height - 2 * margin) / (ymax - ymin)
    def poly(vals: List[float]) -> str:
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    base_vals = [r["baseline"] for r in strict_rows]
    tta_vals = [r["tta"] for r in strict_rows]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for tick in range(6):
        v = ymin + (ymax - ymin) * tick / 5
        yy = y(v)
        parts.append(f'<line x1="{margin-5}" y1="{yy:.1f}" x2="{width-margin}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin-10}" y="{yy+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{100*v:.0f}</text>')
    for i, label in enumerate(labels):
        parts.append(f'<text x="{x(i):.1f}" y="{height-margin+24}" text-anchor="middle" font-size="12" font-family="Arial">{html.escape(label)}</text>')
    parts.append(f'<polyline points="{poly(base_vals)}" fill="none" stroke="#666" stroke-width="3"/>')
    parts.append(f'<polyline points="{poly(tta_vals)}" fill="none" stroke="#d94801" stroke-width="3"/>')
    for i, v in enumerate(base_vals):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="#666"/>')
    for i, v in enumerate(tta_vals):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="#d94801"/>')
    parts.append(f'<rect x="{width-210}" y="55" width="150" height="55" fill="white" stroke="#ddd"/>')
    parts.append(f'<line x1="{width-195}" y1="75" x2="{width-165}" y2="75" stroke="#666" stroke-width="3"/><text x="{width-155}" y="79" font-size="13" font-family="Arial">Baseline</text>')
    parts.append(f'<line x1="{width-195}" y1="98" x2="{width-165}" y2="98" stroke="#d94801" stroke-width="3"/><text x="{width-155}" y="102" font-size="13" font-family="Arial">TTA</text>')
    parts.append('<text x="20" y="215" transform="rotate(-90 20,215)" text-anchor="middle" font-size="13" font-family="Arial">MHR@8 (%)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_histogram(path: Path, values: List[float], title: str, xlabel: str) -> None:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return
    width, height = 760, 420
    margin = 60
    bins = 20
    lo, hi = min(clean), max(clean)
    if lo == hi:
        hi = lo + 1e-6
    counts = [0] * bins
    for value in clean:
        idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_w = (width - 2 * margin) / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for i, count in enumerate(counts):
        x = margin + i * bar_w
        h = count * (height - 2 * margin) / max_count
        y = height - margin - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="#3182bd"/>')
    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="20" y="215" transform="rotate(-90 20,215)" text-anchor="middle" font-size="13" font-family="Arial">Count</text>')
    parts.append(f'<text x="{margin}" y="{height-margin+24}" text-anchor="middle" font-size="11" font-family="Arial">{lo:.3f}</text>')
    parts.append(f'<text x="{width-margin}" y="{height-margin+24}" text-anchor="middle" font-size="11" font-family="Arial">{hi:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def top_doc_titles(record: Dict[str, Any], k: int) -> List[str]:
    return [
        str(doc.get("title", "")) for doc in record.get("documents", [])[:k]
    ]


def gold_evidence(processed_item: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not processed_item:
        return []
    result = []
    for ctx in processed_item.get("contexts", []):
        if ctx.get("is_supporting", False):
            result.append(
                {
                    "title": str(ctx.get("title", "")),
                    "paragraph_text": str(ctx.get("paragraph_text") or ctx.get("text") or ""),
                }
            )
    return result


def select_case_studies(
    base_answer: Dict[str, Dict[str, float]],
    tta_answer: Dict[str, Dict[str, float]],
    base_pq: Dict[str, Dict[str, Any]],
    tta_pq: Dict[str, Dict[str, Any]],
    top_n: int,
) -> List[str]:
    scored = []
    for qid in sorted(set(base_pq) & set(tta_pq)):
        mhr_delta = safe_float(tta_pq[qid].get("final_recall")) - safe_float(base_pq[qid].get("final_recall"))
        f1_delta = tta_answer.get(qid, {}).get("f1", 0.0) - base_answer.get(qid, {}).get("f1", 0.0)
        em_delta = tta_answer.get(qid, {}).get("em", 0.0) - base_answer.get(qid, {}).get("em", 0.0)
        title_delta = safe_float(tta_pq[qid].get("title_only_final_recall")) - safe_float(base_pq[qid].get("title_only_final_recall"))
        if mhr_delta > 0 or f1_delta > 0 or em_delta > 0:
            scored.append((mhr_delta, f1_delta, em_delta, title_delta, qid))
    scored.sort(reverse=True)
    return [qid for *_prefix, qid in scored[:top_n]]


def write_case_studies(
    path: Path,
    qids: List[str],
    base_trace: Dict[str, List[Dict[str, Any]]],
    tta_trace: Dict[str, List[Dict[str, Any]]],
    processed: Dict[str, Dict[str, Any]],
    base_predictions: Dict[str, Any],
    tta_predictions: Dict[str, Any],
    ground_truths: Dict[str, Any],
    base_answer: Dict[str, Dict[str, float]],
    tta_answer: Dict[str, Dict[str, float]],
    top_docs: int,
) -> None:
    lines = ["# Case studies: Baseline vs TTA", ""]
    for idx, qid in enumerate(qids, start=1):
        item = processed.get(qid) or {}
        question = item.get("question_text")
        if not question:
            records = base_trace.get(qid) or tta_trace.get(qid) or []
            question = records[0].get("query", "").strip() if records else qid
        lines.extend(
            [
                f"## Case {idx}: `{qid}`",
                "",
                f"**Question:** {question}",
                "",
                f"**Gold answer:** `{ground_truths.get(qid)}`",
                "",
                f"**Baseline answer:** `{base_predictions.get(qid)}`",
                "",
                f"**TTA answer:** `{tta_predictions.get(qid)}`",
                "",
                "| Metric | Baseline | TTA | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in ["em", "f1", "precision", "recall"]:
            b = base_answer.get(qid, {}).get(metric, 0.0)
            t = tta_answer.get(qid, {}).get(metric, 0.0)
            lines.append(f"| {metric} | {pct(b)} | {pct(t)} | {delta(b, t)} |")
        lines.extend(["", "**Gold evidence:**", ""])
        evidence = gold_evidence(item)
        if evidence:
            for ev in evidence:
                text = ev["paragraph_text"].replace("\n", " ")[:350]
                lines.append(f"- `{ev['title']}`: {text}")
        else:
            lines.append("- Gold evidence was not found in processed data.")
        lines.extend(["", "**Retrieval trajectory:**", ""])
        max_iter = max(len(base_trace.get(qid, [])), len(tta_trace.get(qid, [])))
        for iteration in range(1, max_iter + 1):
            base_record = next((r for r in base_trace.get(qid, []) if r.get("iteration") == iteration), None)
            tta_record = next((r for r in tta_trace.get(qid, []) if r.get("iteration") == iteration), None)
            lines.extend([f"### Iteration {iteration}", ""])
            lines.append("| Rank | Baseline top docs | TTA top docs |")
            lines.append("|---:|---|---|")
            base_titles = top_doc_titles(base_record or {}, top_docs)
            tta_titles = top_doc_titles(tta_record or {}, top_docs)
            for rank in range(top_docs):
                lines.append(
                    f"| {rank + 1} | {base_titles[rank] if rank < len(base_titles) else ''} | {tta_titles[rank] if rank < len(tta_titles) else ''} |"
                )
            if tta_record and tta_record.get("adaptation"):
                adaptation = tta_record["adaptation"]
                l1 = adaptation.get("l1", {})
                l2 = adaptation.get("l2", {})
                lines.extend(
                    [
                        "",
                        f"TTA diagnostics: query_shift={safe_float(adaptation.get('query_shift_l2'), math.nan):.4f}, "
                        f"l1_steps={l1.get('steps', 0)}, "
                        f"l1_loss={l1.get('final_loss', 'n/a')}, "
                        f"l2_loss={l2.get('loss', 'n/a')}",
                        "",
                    ]
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    base_dir: Path,
    tta_dir: Path,
    qa_rows: List[Dict[str, Any]],
    mhr_rows: List[Dict[str, Any]],
    wtl_rows: List[Dict[str, Any]],
    diagnostic_summary: Dict[str, Any],
    outputs: Dict[str, str],
    ablation_rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Baseline vs TTA Analysis Report",
        "",
        f"- Baseline: `{base_dir}`",
        f"- TTA: `{tta_dir}`",
        "",
        "## QA metrics",
        "",
        "| Metric | Baseline | TTA | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in qa_rows:
        lines.append(
            f"| {row['metric']} | {row['baseline_pct']} | {row['tta_pct']} | {row['delta_points']} |"
        )
    lines.extend(["", "## MHR by hop", "", "| Metric | Baseline | TTA | Delta |", "|---|---:|---:|---:|"])
    for row in mhr_rows:
        lines.append(
            f"| {row['metric']} | {row['baseline_pct']} | {row['tta_pct']} | {row['delta_points']} |"
        )
    lines.extend(["", "## Win / Tie / Loss", "", "| Metric | Win | Tie | Loss | Mean delta |", "|---|---:|---:|---:|---:|"])
    for row in wtl_rows:
        lines.append(
            f"| {row['metric']} | {row['win']} | {row['tie']} | {row['loss']} | {row['mean_delta']:.4f} |"
        )
    lines.extend(["", "## TTA diagnostics", ""])
    for key, value in diagnostic_summary.items():
        if isinstance(value, dict):
            clean_value = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in value.items()
            )
            lines.append(f"- `{key}`: {clean_value}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Ablation runs", ""])
    if ablation_rows:
        lines.extend(["| Run | EM | F1 | MHR final | Title-only MHR final | Path |", "|---|---:|---:|---:|---:|---|"])
        for row in ablation_rows:
            lines.append(
                f"| {row['name']} | {row['em_pct']} | {row['f1_pct']} | {row['mhr_pct']} | {row['title_mhr_pct']} | `{row['path']}` |"
            )
    else:
        lines.append("No L1-only/L2-only ablation runs were provided. Run them separately and pass `--ablation_run name=path`.")
    lines.extend(["", "## Generated files", ""])
    for name, output_path in outputs.items():
        lines.append(f"- `{name}`: `{output_path}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_ablation_run(spec: str) -> Dict[str, Any]:
    if "=" not in spec:
        raise ValueError("--ablation_run must use name=path format")
    name, raw_path = spec.split("=", 1)
    run_dir = Path(raw_path)
    eval_path = run_dir / "test_evaluation.json"
    retrieval_path = run_dir / "test_retrieval_evaluation.json"
    row = {"name": name, "path": str(run_dir)}
    if eval_path.exists():
        data = read_json(eval_path)
        row.update(
            {
                "em": safe_float(data.get("em")),
                "f1": safe_float(data.get("f1")),
                "em_pct": pct(safe_float(data.get("em"))),
                "f1_pct": pct(safe_float(data.get("f1"))),
            }
        )
    else:
        row.update({"em": math.nan, "f1": math.nan, "em_pct": "n/a", "f1_pct": "n/a"})
    if retrieval_path.exists():
        data = read_json(retrieval_path)
        mhr_key = next((k for k in data if k.startswith("MHR_final@")), None)
        title_key = next((k for k in data if k.startswith("title_only_MHR_final@")), None)
        row.update(
            {
                "mhr": safe_float(data.get(mhr_key)) if mhr_key else math.nan,
                "title_mhr": safe_float(data.get(title_key)) if title_key else math.nan,
                "mhr_pct": pct(safe_float(data.get(mhr_key))) if mhr_key else "n/a",
                "title_mhr_pct": pct(safe_float(data.get(title_key))) if title_key else "n/a",
            }
        )
    else:
        row.update({"mhr": math.nan, "title_mhr": math.nan, "mhr_pct": "n/a", "title_mhr_pct": "n/a"})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze baseline vs TTA inference outputs.")
    parser.add_argument("--baseline_dir", required=True, type=Path)
    parser.add_argument("--tta_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--case_count", type=int, default=8)
    parser.add_argument("--top_docs", type=int, default=8)
    parser.add_argument("--tie_epsilon", type=float, default=1e-9)
    parser.add_argument(
        "--ablation_run",
        action="append",
        default=[],
        help="Optional extra run in name=prediction_dir format, e.g. l1=/path/to/best",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_eval = read_json(args.baseline_dir / "test_evaluation.json")
    tta_eval = read_json(args.tta_dir / "test_evaluation.json")
    base_retrieval_eval = read_json(args.baseline_dir / "test_retrieval_evaluation.json")
    tta_retrieval_eval = read_json(args.tta_dir / "test_retrieval_evaluation.json")
    base_predictions = read_json(args.baseline_dir / "test_prediction.json")
    tta_predictions = read_json(args.tta_dir / "test_prediction.json")
    ground_truths = read_json(args.tta_dir / "test_ground_truth.json")

    base_trace = load_trace_by_qid(args.baseline_dir / "test_retrieval_trace.jsonl")
    tta_trace = load_trace_by_qid(args.tta_dir / "test_retrieval_trace.jsonl")
    base_pq = load_per_question(args.baseline_dir / "test_retrieval_per_question.jsonl")
    tta_pq = load_per_question(args.tta_dir / "test_retrieval_per_question.jsonl")

    dataset = infer_dataset_from_config(args.tta_dir) or infer_dataset_from_config(args.baseline_dir)
    processed = load_processed_data(dataset)

    qa_rows = []
    for metric in ["em", "f1", "precision", "recall"]:
        base = safe_float(base_eval.get(metric))
        tta = safe_float(tta_eval.get(metric))
        qa_rows.append(
            {
                "metric": metric,
                "baseline": base,
                "tta": tta,
                "delta": tta - base,
                "baseline_pct": pct(base),
                "tta_pct": pct(tta),
                "delta_points": delta(base, tta),
            }
        )
    write_csv(args.output_dir / "qa_metrics.csv", qa_rows, list(qa_rows[0].keys()))

    mhr_rows = extract_mhr_rows(base_retrieval_eval, tta_retrieval_eval)
    write_csv(args.output_dir / "mhr_by_hop.csv", mhr_rows, list(mhr_rows[0].keys()))
    svg_line_chart(args.output_dir / "mhr_by_hop.svg", mhr_rows, "MHR@8 by Retrieval Hop")

    base_answer = per_question_answer_metrics(base_predictions, ground_truths)
    tta_answer = per_question_answer_metrics(tta_predictions, ground_truths)
    wtl_rows = build_win_tie_loss(
        base_answer, tta_answer, base_pq, tta_pq, args.tie_epsilon
    )
    write_csv(args.output_dir / "win_tie_loss.csv", wtl_rows, list(wtl_rows[0].keys()))

    coverage_rows = []
    coverage_rows.extend(evidence_coverage_rows("baseline", base_pq, title_only=False))
    coverage_rows.extend(evidence_coverage_rows("tta", tta_pq, title_only=False))
    coverage_rows.extend(evidence_coverage_rows("baseline", base_pq, title_only=True))
    coverage_rows.extend(evidence_coverage_rows("tta", tta_pq, title_only=True))
    write_csv(
        args.output_dir / "evidence_coverage.csv",
        coverage_rows,
        ["setting", "matched_gold_evidence", "question_count", "question_rate", "title_only"],
    )

    diagnostics, trend_rows = collect_tta_diagnostics(tta_trace)
    if diagnostics:
        write_csv(args.output_dir / "tta_diagnostics_by_hop.csv", diagnostics, list(diagnostics[0].keys()))
    if trend_rows:
        fieldnames = sorted({key for row in trend_rows for key in row})
        write_csv(args.output_dir / "tta_loss_trend_by_iteration.csv", trend_rows, fieldnames)
    query_shift_values = [safe_float(row.get("query_shift_l2"), math.nan) for row in diagnostics]
    svg_histogram(
        args.output_dir / "query_shift_distribution.svg",
        query_shift_values,
        "TTA Query Shift Distribution",
        "query_shift_l2",
    )
    l1_values = [safe_float(row.get("l1_loss"), math.nan) for row in diagnostics]
    l2_values = [safe_float(row.get("l2_loss"), math.nan) for row in diagnostics]
    diagnostic_summary = {
        "trace_records": len(diagnostics),
        "unique_questions": len(tta_trace),
        "pseudo_ok_rate": (
            sum(1 for row in diagnostics if row.get("pseudo_label_ok")) / len(diagnostics)
            if diagnostics else 0.0
        ),
        "query_shift_l2": summary_stats(query_shift_values),
        "l1_loss": summary_stats(l1_values),
        "l2_loss": summary_stats(l2_values),
        "mean_l1_steps": mean([int(row.get("l1_steps") or 0) for row in diagnostics]) if diagnostics else 0.0,
    }
    write_json(args.output_dir / "tta_diagnostic_summary.json", diagnostic_summary)

    selected_qids = select_case_studies(
        base_answer, tta_answer, base_pq, tta_pq, args.case_count
    )
    write_case_studies(
        args.output_dir / "case_studies.md",
        selected_qids,
        base_trace,
        tta_trace,
        processed,
        base_predictions,
        tta_predictions,
        ground_truths,
        base_answer,
        tta_answer,
        args.top_docs,
    )

    ablation_rows = [summarize_ablation_run(spec) for spec in args.ablation_run]
    if ablation_rows:
        write_csv(
            args.output_dir / "ablation_summary.csv",
            ablation_rows,
            ["name", "em_pct", "f1_pct", "mhr_pct", "title_mhr_pct", "path"],
        )

    outputs = {
        "qa_metrics": str(args.output_dir / "qa_metrics.csv"),
        "mhr_by_hop": str(args.output_dir / "mhr_by_hop.csv"),
        "mhr_by_hop_plot": str(args.output_dir / "mhr_by_hop.svg"),
        "win_tie_loss": str(args.output_dir / "win_tie_loss.csv"),
        "query_shift_distribution": str(args.output_dir / "query_shift_distribution.svg"),
        "tta_loss_trend": str(args.output_dir / "tta_loss_trend_by_iteration.csv"),
        "evidence_coverage": str(args.output_dir / "evidence_coverage.csv"),
        "case_studies": str(args.output_dir / "case_studies.md"),
        "tta_diagnostic_summary": str(args.output_dir / "tta_diagnostic_summary.json"),
    }
    write_report(
        args.output_dir / "report.md",
        args.baseline_dir,
        args.tta_dir,
        qa_rows,
        mhr_rows,
        wtl_rows,
        diagnostic_summary,
        outputs,
        ablation_rows,
    )
    print(f"Analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
