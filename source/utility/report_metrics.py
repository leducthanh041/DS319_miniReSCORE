import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, Optional


class SectionTimer:
    def __init__(self):
        self._starts: Dict[str, float] = {}
        self.sections: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self._starts:
            return 0.0
        elapsed = time.perf_counter() - self._starts.pop(name)
        self.sections[name] = self.sections.get(name, 0.0) + elapsed
        return elapsed


def count_model_parameters(model: Any) -> Dict[str, Any]:
    if model is None or not hasattr(model, "parameters"):
        return {
            "available": False,
            "reason": "model_parameters_not_available_in_current_process",
        }

    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    return {
        "available": True,
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 6),
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _round_or_none(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


def build_latency_metrics(total_sec: float, num_examples: int) -> Dict[str, Any]:
    latency_per_question = _safe_ratio(total_sec, num_examples)
    throughput_qps = _safe_ratio(num_examples, total_sec)
    return {
        "measurement_scope": (
            "controller.run only; excludes model/index loading and post-hoc evaluation"
        ),
        "num_examples": num_examples,
        "total_time_sec": round(total_sec, 6),
        "total_time_hours": round(total_sec / 3600.0, 6),
        "total_time_days": round(total_sec / 86400.0, 6),
        "latency_per_question_sec": _round_or_none(latency_per_question),
        "throughput_questions_per_sec": _round_or_none(throughput_qps),
        "throughput_questions_per_hour": (
            _round_or_none(throughput_qps * 3600.0) if throughput_qps is not None else None
        ),
    }


def build_quality_metrics(
    evaluation_results: Dict[str, Any],
    official_evaluation_results: Dict[str, Any],
    retrieval_evaluation_results: Dict[str, Any],
    retrieval_count: int,
) -> Dict[str, Any]:
    source = official_evaluation_results or evaluation_results or {}
    return {
        "em": _safe_float(source.get("em")),
        "f1": _safe_float(source.get("f1")),
        "precision": _safe_float(source.get("precision")),
        "recall": _safe_float(source.get("recall")),
        "sp_em": _safe_float(source.get("sp_em")),
        "sp_f1": _safe_float(source.get("sp_f1")),
        "sp_precision": _safe_float(source.get("sp_precision")),
        "sp_recall": _safe_float(source.get("sp_recall")),
        f"MHR_final@{retrieval_count}": _safe_float(
            retrieval_evaluation_results.get(f"MHR_final@{retrieval_count}")
        ),
        f"title_only_MHR_final@{retrieval_count}": _safe_float(
            retrieval_evaluation_results.get(f"title_only_MHR_final@{retrieval_count}")
        ),
        "mhr_count": retrieval_evaluation_results.get("count"),
    }


def summarize_tta_behavior(trace_file_path: str) -> Dict[str, Any]:
    if not trace_file_path or not os.path.exists(trace_file_path):
        return {
            "available": False,
            "reason": "retrieval_trace_not_found",
            "trace_file": trace_file_path,
        }

    records = []
    with open(trace_file_path, "r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {
            "available": False,
            "reason": "retrieval_trace_empty",
            "trace_file": trace_file_path,
        }

    by_question = defaultdict(int)
    l1_steps = []
    l1_losses = []
    l2_losses = []
    query_shifts = []
    l2_updates = 0
    pseudo_ok = 0

    for record in records:
        by_question[str(record.get("question_id"))] += 1
        adaptation = record.get("adaptation", {}) or {}
        l1 = adaptation.get("l1", {}) or {}
        l2 = adaptation.get("l2", {}) or {}

        if record.get("pseudo_label_ok", False):
            pseudo_ok += 1
        if "steps" in l1:
            l1_steps.append(float(l1.get("steps") or 0))
        if l1.get("loss") is not None:
            l1_losses.append(float(l1["loss"]))
        if l2.get("loss") is not None:
            l2_losses.append(float(l2["loss"]))
        if adaptation.get("query_shift_l2") is not None:
            query_shifts.append(float(adaptation["query_shift_l2"]))
        if l2.get("updated", False):
            l2_updates += 1

    def avg(values):
        return round(sum(values) / len(values), 6) if values else None

    return {
        "available": True,
        "trace_file": trace_file_path,
        "retrieval_hops": len(records),
        "unique_questions": len(by_question),
        "avg_retrieval_hops_per_question": avg(list(by_question.values())),
        "pseudo_label_success_rate": round(pseudo_ok / len(records), 6),
        "pseudo_label_failure_count": len(records) - pseudo_ok,
        "avg_l1_steps_per_hop": avg(l1_steps),
        "avg_l1_loss": avg(l1_losses),
        "avg_l2_loss": avg(l2_losses),
        "avg_query_shift_l2": avg(query_shifts),
        "median_query_shift_l2": (
            round(statistics.median(query_shifts), 6) if query_shifts else None
        ),
        "l2_update_count": l2_updates,
    }


def _quality_gain(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    gains = {}
    for key, value in current.items():
        current_value = _safe_float(value)
        baseline_value = _safe_float(baseline.get(key))
        if current_value is None or baseline_value is None:
            continue
        abs_gain = current_value - baseline_value
        rel_gain = _safe_ratio(abs_gain, baseline_value)
        gains[f"{key}_abs_gain"] = round(abs_gain, 6)
        gains[f"{key}_relative_gain_pct"] = (
            round(rel_gain * 100.0, 6) if rel_gain is not None else None
        )
    return gains


def load_report_metrics(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline report metrics not found: {path}")
    with open(path, "r", encoding="utf-8") as report_file:
        return json.load(report_file)


def build_report_metrics(
    *,
    cfg: Any,
    evaluation_results: Dict[str, Any],
    official_evaluation_results: Dict[str, Any],
    retrieval_evaluation_results: Dict[str, Any],
    num_examples: int,
    pipeline_time_sec: float,
    section_times: Dict[str, float],
    retriever: Any = None,
    generator: Any = None,
    cross_encoder: Any = None,
    lora_stats: Optional[Dict[str, Any]] = None,
    tta_trace_file_path: Optional[str] = None,
    tta_variant: Optional[str] = None,
    baseline_report_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime = build_latency_metrics(pipeline_time_sec, num_examples)
    quality = build_quality_metrics(
        evaluation_results,
        official_evaluation_results,
        retrieval_evaluation_results,
        cfg.retrieval_count,
    )

    parameters = {
        "retriever_query_encoder": count_model_parameters(
            getattr(retriever, "query_model", None)
        ),
        "cross_encoder": count_model_parameters(
            getattr(cross_encoder, "model", None)
        ),
        "generator": {
            "available": False,
            "reason": (
                "vllm_server_or_external_generator_not_counted"
                if generator is not None
                else "generator_not_available"
            ),
            "model_name": getattr(cfg, "generation_model_name", None),
        },
        "lora": lora_stats or {
            "available": False,
            "reason": "lora_not_enabled_or_not_counted",
        },
    }

    report = {
        "schema_version": 1,
        "run": {
            "running_name": cfg.running_name,
            "dataset": cfg.dataset,
            "dataset_split": cfg.dataset_split,
            "method": cfg.method,
            "tta_variant": tta_variant,
            "retrieval_count": cfg.retrieval_count,
            "prediction_dir": cfg.prediction_file_dir,
        },
        "quality": quality,
        "runtime": runtime,
        "runtime_sections_sec": {
            key: round(value, 6) for key, value in section_times.items()
        },
        "parameters": parameters,
        "cost_effectiveness": {},
    }

    if tta_trace_file_path:
        report["tta_behavior"] = summarize_tta_behavior(tta_trace_file_path)

    if baseline_report_metrics:
        baseline_quality = baseline_report_metrics.get("quality", {})
        baseline_runtime = baseline_report_metrics.get("runtime", {})
        quality_gain = _quality_gain(quality, baseline_quality)
        baseline_total_days = _safe_float(baseline_runtime.get("total_time_days"))
        current_total_days = _safe_float(runtime.get("total_time_days"))
        current_total_sec = _safe_float(runtime.get("total_time_sec"))
        baseline_total_sec = _safe_float(baseline_runtime.get("total_time_sec"))

        report["quality_gain_vs_baseline"] = quality_gain
        report["runtime"]["overhead_vs_baseline"] = _round_or_none(
            _safe_ratio(current_total_sec, baseline_total_sec)
        )
        for key, value in quality_gain.items():
            if not key.endswith("_abs_gain"):
                continue
            gain = _safe_float(value)
            metric_name = key[: -len("_abs_gain")]
            report["cost_effectiveness"][f"{metric_name}_gain_per_tta_day"] = (
                _round_or_none(_safe_ratio(gain, current_total_days))
            )
            report["cost_effectiveness"][f"{metric_name}_gain_per_extra_day"] = (
                _round_or_none(
                    _safe_ratio(
                        gain,
                        current_total_days - baseline_total_days
                        if current_total_days is not None and baseline_total_days is not None
                        else None,
                    )
                )
            )

    return report


def write_report_metrics(report: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=4)
