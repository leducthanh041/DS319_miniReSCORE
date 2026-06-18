# From IRCoT

import re
import os
import json
import uuid
import subprocess
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from source.evaluation.lib import (
    get_retriever_address,
    get_llm_server_address,
    infer_source_target_prefix,
    infer_dataset_from_file_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    get_config_file_path_from_name_or_path,
)
from source.evaluation.metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from source.evaluation.metrics.support_em_f1 import SupportEmF1Metric
from source.evaluation.metrics.answer_support_recall import AnswerSupportRecallMetric


def _fallback_official_evaluation(
    reason: str,
    prediction_type: str,
    id_to_ground_truths: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
) -> Dict:
    print(
        "[warning] Official evaluation is unavailable; "
        f"falling back to internal evaluation. Reason: {reason}"
    )
    metrics = evaluate_by_dicts(prediction_type, id_to_ground_truths, id_to_predictions)
    metrics["official_evaluation_skipped"] = True
    metrics["official_evaluation_skip_reason"] = reason
    return metrics


def answer_extractor(
    potentially_cot: str
) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output


def _normalize_retrieval_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _document_key(title: Any, paragraph_text: Any) -> Tuple[str, str]:
    return (
        _normalize_retrieval_text(title),
        _normalize_retrieval_text(paragraph_text),
    )


def _title_key(title: Any) -> str:
    return _normalize_retrieval_text(title)


def _normalize_retrieval_doc(title: Any, paragraph_text: Any) -> Dict[str, str]:
    return {
        "title": _normalize_retrieval_text(title),
        "paragraph_text": _normalize_retrieval_text(paragraph_text),
    }


def _gold_support_docs(contexts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], set]:
    strict_docs = []
    title_keys = set()

    for context in contexts:
        if not context.get("is_supporting", False):
            continue
        title = context.get("title")
        paragraph_text = context.get("paragraph_text") or context.get("text")
        strict_docs.append(_normalize_retrieval_doc(title, paragraph_text))
        title_keys.add(_title_key(title))

    return strict_docs, title_keys


def _retrieved_docs(documents: List[Dict[str, Any]], k: int) -> Tuple[List[Dict[str, str]], set]:
    strict_docs_by_key = {}
    title_keys = set()

    for document in documents[:k]:
        title = document.get("title")
        paragraph_text = document.get("paragraph_text") or document.get("text")
        doc = _normalize_retrieval_doc(title, paragraph_text)
        strict_docs_by_key[_document_key(title, paragraph_text)] = doc
        title_keys.add(_title_key(title))

    return list(strict_docs_by_key.values()), title_keys


def _documents_match(gold_doc: Dict[str, str], retrieved_doc: Dict[str, str]) -> bool:
    if gold_doc["title"] != retrieved_doc["title"]:
        return False

    gold_text = gold_doc["paragraph_text"]
    retrieved_text = retrieved_doc["paragraph_text"]
    if not gold_text or not retrieved_text:
        return False

    return (
        gold_text == retrieved_text
        or gold_text in retrieved_text
        or retrieved_text in gold_text
    )


def _support_docs_recall(retrieved_docs: List[Dict[str, str]], gold_docs: List[Dict[str, str]]) -> float:
    if not gold_docs:
        return 0.0

    matched_count = 0
    for gold_doc in gold_docs:
        if any(_documents_match(gold_doc, retrieved_doc) for retrieved_doc in retrieved_docs):
            matched_count += 1

    return matched_count / float(len(gold_docs))


def _safe_recall(retrieved_keys: set, gold_keys: set) -> float:
    if not gold_keys:
        return 0.0
    return len(retrieved_keys & gold_keys) / float(len(gold_keys))


def evaluate_multi_hop_recall_at_k(
    contexts_by_qid: Dict[str, List[Dict[str, Any]]],
    retrieval_trace_file_path: str,
    k: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate cumulative multi-hop recall over inference retrieval traces.

    MHR_i@k is computed against gold supporting contexts and accumulates the
    top-k retrieved documents from iteration 1 through i.
    """
    traces_by_qid = defaultdict(list)
    if not os.path.exists(retrieval_trace_file_path):
        raise FileNotFoundError(f"Retrieval trace file not found: {retrieval_trace_file_path}")

    with open(retrieval_trace_file_path, "r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record.get("question_id"))
            record["iteration"] = int(record.get("iteration", 0))
            traces_by_qid[qid].append(record)

    per_question = []
    max_iteration_seen = 0

    for qid, contexts in contexts_by_qid.items():
        gold_strict, gold_titles = _gold_support_docs(contexts)
        if not gold_strict:
            continue

        records = sorted(
            traces_by_qid.get(str(qid), []),
            key=lambda item: item.get("iteration", 0),
        )
        if records:
            max_iteration_seen = max(max_iteration_seen, max(record["iteration"] for record in records))

        retrieved_strict_so_far = {}
        retrieved_titles_so_far = set()
        strict_recall_by_iteration = {}
        title_recall_by_iteration = {}

        last_iteration = 0
        for record in records:
            iteration = record["iteration"]
            strict_docs, title_keys = _retrieved_docs(record.get("documents", []), k)
            for document in strict_docs:
                retrieved_strict_so_far[(document["title"], document["paragraph_text"])] = document
            retrieved_titles_so_far.update(title_keys)

            strict_recall_by_iteration[str(iteration)] = _support_docs_recall(
                list(retrieved_strict_so_far.values()),
                gold_strict,
            )
            title_recall_by_iteration[str(iteration)] = _safe_recall(retrieved_titles_so_far, gold_titles)
            last_iteration = iteration

        final_strict_recall = _support_docs_recall(
            list(retrieved_strict_so_far.values()),
            gold_strict,
        )
        final_title_recall = _safe_recall(retrieved_titles_so_far, gold_titles)
        per_question.append(
            {
                "question_id": qid,
                "num_gold_supports": len(gold_strict),
                "num_retrieval_iterations": last_iteration,
                "recall_by_iteration": strict_recall_by_iteration,
                "final_recall": final_strict_recall,
                "title_only_recall_by_iteration": title_recall_by_iteration,
                "title_only_final_recall": final_title_recall,
            }
        )

    count = len(per_question)
    metrics: Dict[str, Any] = {
        "k": k,
        "count": count,
        "retrieval_trace_file": retrieval_trace_file_path,
        "matching": "title_match_and_paragraph_exact_or_contains",
    }

    for iteration in range(1, max_iteration_seen + 1):
        values = []
        title_values = []
        for item in per_question:
            recall_by_iteration = item["recall_by_iteration"]
            title_recall_by_iteration = item["title_only_recall_by_iteration"]
            available_iterations = sorted(int(idx) for idx in recall_by_iteration.keys())
            latest_iteration = max(
                [idx for idx in available_iterations if idx <= iteration],
                default=None,
            )
            if latest_iteration is None:
                values.append(0.0)
                title_values.append(0.0)
            else:
                values.append(recall_by_iteration[str(latest_iteration)])
                title_values.append(title_recall_by_iteration[str(latest_iteration)])

        metrics[f"MHR_{iteration}@{k}"] = round(sum(values) / count, 6) if count else 0.0
        metrics[f"title_only_MHR_{iteration}@{k}"] = (
            round(sum(title_values) / count, 6) if count else 0.0
        )

    final_values = [item["final_recall"] for item in per_question]
    final_title_values = [item["title_only_final_recall"] for item in per_question]
    metrics[f"MHR_final@{k}"] = round(sum(final_values) / count, 6) if count else 0.0
    metrics[f"title_only_MHR_final@{k}"] = (
        round(sum(final_title_values) / count, 6) if count else 0.0
    )

    return metrics, per_question


def evaluate_by_dicts(
    prediction_type: str,
    id_to_ground_truths: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
) -> Dict:
    
    if prediction_type == "answer":
        metrics = [DropAnswerEmAndF1(), SupportEmF1Metric(do_normalize_answer=True)]
    elif prediction_type in ("titles", "pids", "real_pids"):
        metrics = [SupportEmF1Metric()]
    elif prediction_type in ("paras"):
        metrics = [AnswerSupportRecallMetric()]

    for id_ in set(id_to_ground_truths.keys()):
        ground_truth = id_to_ground_truths[id_]
        prediction = id_to_predictions[id_]

        assert isinstance(prediction, (str, list))
        if prediction_type == "answer" and isinstance(prediction, str):
            if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
            else:
                prediction = [prediction]

        assert isinstance(prediction, (list, tuple))
        prediction = [str(e) for e in prediction]

        if prediction_type == "answer":
            prediction = [answer_extractor(_prediction) for _prediction in prediction]  # Temporary.
            metrics[0](prediction, [ground_truth])
            metrics[1](prediction, ground_truth)
        elif prediction_type in ("titles", "pids", "real_pids"):
            metrics[0](prediction, ground_truth)
        elif prediction_type in ("paras"):
            predicted_paras = [
                " ".join([eval(prediction_)["title"], eval(prediction_)["paragraph_text"]])
                for prediction_ in prediction
            ]
            metrics[0](predicted_paras, ground_truth)

    evaluation_results = metrics[0].get_metric()
    if prediction_type == "answer":
        evaluation_results_ = metrics[1].get_metric()
        evaluation_results["sp_em"] = evaluation_results_["title_em"]
        evaluation_results["sp_f1"] = evaluation_results_["title_f1"]
        evaluation_results["sp_precision"] = evaluation_results_["title_precision"]
        evaluation_results["sp_recall"] = evaluation_results_["title_recall"]

    return evaluation_results


def official_evaluate_by_dicts(
    prediction_type: str, 
    id_to_predictions: Dict[str, Any], 
    id_to_ground_truths: Dict[str, Any], 
    dataset: str
) -> Dict:

    if prediction_type != "answer":
        # official evaluation is not available for non answer prediction.
        return evaluate_by_dicts(prediction_type, id_to_ground_truths, id_to_predictions)

    question_ids = list(id_to_predictions.keys())

    for id_, prediction in id_to_predictions.items():
        if isinstance(prediction, list) and len(prediction) == 1:
            id_to_predictions[id_] = str(prediction[0])
        elif isinstance(prediction, list) and len(prediction) > 1:
            id_to_predictions[id_] = " ".join([str(e) for e in prediction])
            print("WARNING: Found a list answer prediction, concatenating it.")

    os.makedirs(".temp", exist_ok=True)

    official_evaluator_scripts = {
        "hotpotqa": os.path.join(
            "source", "evaluation", "official_evaluation", "hotpotqa", "hotpot_evaluate_v1.py"
        ),
        "2wikimultihopqa": os.path.join(
            "source", "evaluation", "official_evaluation", "2wikimultihopqa", "2wikimultihop_evaluate_v1.1.py"
        ),
        "musique": os.path.join(
            "source", "evaluation", "official_evaluation", "musique", "evaluate_v1.0.py"
        ),
    }
    official_evaluator_script = official_evaluator_scripts.get(dataset)
    if official_evaluator_script and not os.path.exists(official_evaluator_script):
        return _fallback_official_evaluation(
            reason=f"missing official evaluator script: {official_evaluator_script}",
            prediction_type=prediction_type,
            id_to_ground_truths=id_to_ground_truths,
            id_to_predictions=id_to_predictions,
        )

    if dataset == "hotpotqa":
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_json(os.path.join("data", "raw_data", "hotpotqa", "hotpot_dev_distractor_v1.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        # Run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_prediction_file_path)
        temp_output_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        official_hotpotqa_evaluation_path = os.path.join("source", "evaluation", "official_evaluation", "hotpotqa")
        command = (
            f"cd {official_hotpotqa_evaluation_path} ; "
            + f"python hotpot_evaluate_v1.py {temp_prediction_file_path} "
            + f"{temp_ground_truth_file_path} > {temp_output_file_path}"
        )
        status = subprocess.call(command, shell=True)
        if status != 0:
            raise Exception("Running the official evaluation script failed.")

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, "")
        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = eval(file.read().strip())
            metrics = {
                "f1": round(metrics_["f1"], 3),
                "em": round(metrics_["em"], 3),
                "precision": round(metrics_["prec"], 3),
                "recall": round(metrics_["recall"], 3),
                "count": len(id_to_predictions),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "2wikimultihopqa":
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_json(os.path.join("data", "raw_data", "2wikimultihopqa", "dev.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
            "evidence": {id_: ["", "", ""] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        # run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_prediction_file_path)
        alias_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, "data", "raw_data", "2wikimultihopqa", "id_aliases.json")
        temp_output_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        evaluation_directory = os.path.join("source", "evaluation", "official_evaluation", "2wikimultihopqa")
        command = (
            f"cd {evaluation_directory} ; "
            + f"python 2wikimultihop_evaluate_v1.1.py {temp_prediction_file_path} "
            + f"{temp_ground_truth_file_path} {alias_file_path} > {temp_output_file_path}"
        )
        subprocess.call(command, shell=True)

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, "")
        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = json.loads(file.read().strip())
            metrics = {
                "f1": round(metrics_["f1"] / 100, 3),
                "em": round(metrics_["em"] / 100, 3),
                "precision": round(metrics_["prec"] / 100, 3),
                "recall": round(metrics_["recall"] / 100, 3),
                "count": len(id_to_predictions),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "musique":
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_jsonl(os.path.join("data", "raw_data", "musique", "musique_ans_v1.0_dev.jsonl"))
        original_keyed_data = {datum["id"]: datum for datum in original_data}
        filtered_data = [original_keyed_data[qid] for qid in question_ids]
        write_jsonl(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = [
            {
                "id": id_,
                "predicted_answer": str(id_to_predictions[id_]),
                "predicted_support_idxs": [0, 1],
                "predicted_answerable": True,
            }
            for id_ in question_ids
        ]
        write_jsonl(data, temp_prediction_file_path)

        # run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, temp_prediction_file_path)
        temp_output_file_path = os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        evaluation_directory = os.path.join("source", "evaluation", "official_evaluation", "musique")
        command = (
            f"cd {evaluation_directory} ; "
            + f"python evaluate_v1.0.py {temp_prediction_file_path} {temp_ground_truth_file_path} "
            + f"--output_filepath {temp_output_file_path}"
        )
        subprocess.call(command, shell=True)

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir) + os.path.sep, "")
        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = json.loads(file.read().strip())
            metrics = {
                "f1": round(metrics_["answer_f1"], 3),
                "em": round(metrics_["answer_em"], 3) if "answer_em" in metrics_ else None,
                "count": len(id_to_predictions),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "iirc":
        return evaluate_by_dicts("answer", id_to_ground_truths, id_to_predictions)


def load_predictions(prediction_file_path: str) -> Dict:
    with open(prediction_file_path, "r") as file:
        id_to_predictions = json.load(file)
    return id_to_predictions


















def main():
    parser = argparse.ArgumentParser(description="Run source.evaluation.")
    parser.add_argument("experiment_name_or_path", type=str, help="experiment_name_or_path")
    parser.add_argument("evaluation_path", type=str, help="evaluation_path")
    # parser.add_argument("--prediction-type", type=str, help="optional prediction-type", choices=PREDICTION_TYPES)
    parser.add_argument(
        "--prediction-suffix", type=str, help="optional suffix for the prediction directory.", default=""
    )
    parser.add_argument(
        "--question-type-key-value", type=str, help="':' separated question-type-key-value.", default=None
    )
    parser.add_argument("--only-print", action="store_true", default=False, help="only print don't run evaluation")
    parser.add_argument(
        "--official", action="store_true", default=False, help="use official eval scripts when available."
    )

    args = parser.parse_args()

    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]

    prediction_directory = os.path.join("predictions", experiment_name + args.prediction_suffix)
    prediction_file_name = os.path.splitext(os.path.basename(args.evaluation_path))[0]
    prediction_file_name = infer_source_target_prefix(config_filepath, args.evaluation_path) + prediction_file_name
    prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")

    if not os.path.exists(prediction_file_path):
        exit(f"The prediction_file_path {prediction_file_path} is not available.")

    official_prefix = "official_" if args.official else ""
    save_metrics_path = os.path.join(
        prediction_directory, official_prefix + "evaluation_metrics__" + prediction_file_name + ".json"
    )
    if args.only_print:
        if not os.path.exists(save_metrics_path):
            exit("Asked to print the metrics, but the metrics file_path is not available.")
        with open(save_metrics_path, "r") as file:
            print(file.read())
        exit()

    # get prediction_type
    experiment_config = load_experiment_config(config_filepath)
    prediction_type = experiment_config["prediction_type"]

    # prep ground_truths
    question_type_key = question_type_value = None
    if args.question_type_key_value is not None:
        if args.question_type_key_value.count(":") != 1:
            raise Exception("The question_type_key_value must be : separated.")
        question_type_key, question_type_value = args.question_type_key_value.split(":")
        question_type_key = question_type_key.strip()
        question_type_value = question_type_value.strip()

    id_to_ground_truths = load_ground_truths(
        experiment_config,
        args.evaluation_path,
        question_type_key=question_type_key,
        question_type_value=question_type_value,
    )

    # prep predictions
    id_to_predictions = load_predictions(prediction_file_path)

    if question_type_value is not None:
        id_to_predictions = {
            id_: prediction for id_, prediction in id_to_predictions.items() if id_ in id_to_ground_truths.keys()
        }

    # verify equality
    if set(id_to_ground_truths.keys()) != set(id_to_predictions.keys()):
        exit("Ids in input examples and predictions don't match.")

    # evaluate
    if args.official:
        dataset = infer_dataset_from_file_path(args.evaluation_path)
        evaluation_results = official_evaluate_by_dicts(
            prediction_type=prediction_type,
            id_to_predictions=id_to_predictions,
            id_to_ground_truths=id_to_ground_truths,
            dataset=dataset,
        )
    else:
        evaluation_results = evaluate_by_dicts(
            prediction_type=prediction_type,
            id_to_predictions=id_to_predictions,
            id_to_ground_truths=id_to_ground_truths,
        )
    print(json.dumps(evaluation_results, indent=4))

    # To be able to reproduce the same result, save git-hash
    git_hash_filepath = os.path.join(prediction_directory, "git_hash__" + prediction_file_name + ".txt")
    if os.path.exists(git_hash_filepath):
        with open(git_hash_filepath, "r") as file:
            git_hash = file.read().strip()
        evaluation_results["git_hash"] = git_hash

    # Save the evaluation metrics
    print(f"Saving metrics in {save_metrics_path}")
    with open(save_metrics_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)
    print(evaluation_results)

    # Save the ground_truth used in the same json/dict format (just for convenience)
    ground_truth_in_dict_file_path = os.path.join(
        prediction_directory, "ground_truth__" + prediction_file_name + ".json"
    )
    with open(ground_truth_in_dict_file_path, "w") as file:
        json.dump(id_to_ground_truths, file, indent=4)


if __name__ == "__main__":
    main()
