import json
import os
from collections import Counter
from typing import Dict, List

from tqdm import tqdm


def write_hotpotqa_instances_to_filepath(instances: List[Dict], full_filepath: str):
    max_num_tokens = 1000
    hop_sizes = Counter()

    print(f"Writing in: {full_filepath}")
    with open(full_filepath, "w", encoding="utf-8") as full_file:
        for raw_instance in tqdm(instances):
            processed_instance = {}
            processed_instance["dataset"] = "hotpotqa"
            processed_instance["question_id"] = raw_instance["_id"]
            processed_instance["question_text"] = raw_instance["question"]
            processed_instance["level"] = raw_instance["level"]
            processed_instance["type"] = raw_instance["type"]

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            processed_instance["answers_objects"] = [answers_object]

            raw_context = raw_instance["context"]
            supporting_titles = [title for title, _ in raw_instance["supporting_facts"]]

            title_to_paragraph = {
                title: "".join(text)
                for title, text in raw_context
            }
            paragraph_to_title = {
                "".join(text): title
                for title, text in raw_context
            }

            gold_paragraph_texts = [title_to_paragraph[title] for title in supporting_titles if title in title_to_paragraph]
            gold_paragraph_texts = set(gold_paragraph_texts)

            paragraph_texts = ["".join(paragraph) for _, paragraph in raw_context]
            paragraph_texts = list(set(paragraph_texts))

            processed_instance["contexts"] = [
                {
                    "idx": index,
                    "title": paragraph_to_title[paragraph_text].strip(),
                    "paragraph_text": " ".join(paragraph_text.strip().split(" ")[:max_num_tokens]),
                    "is_supporting": paragraph_text in gold_paragraph_texts,
                }
                for index, paragraph_text in enumerate(paragraph_texts)
            ]

            supporting_contexts = [context for context in processed_instance["contexts"] if context["is_supporting"]]
            hop_sizes[len(supporting_contexts)] += 1

            full_file.write(json.dumps(processed_instance, ensure_ascii=False) + "\n")

    print(f"Hop-sizes: {str(hop_sizes)}")


def read_hotpotqa_instances(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    directory = os.path.join("data", "processed_data", "hotpotqa")
    os.makedirs(directory, exist_ok=True)

    raw_train_path = os.path.join("data", "raw_data", "hotpotqa", "hotpot_train_v1.1.json")
    raw_dev_path = os.path.join("data", "raw_data", "hotpotqa", "hotpot_dev_distractor_v1.json")

    processed_train_filepath = os.path.join(directory, "train.jsonl")
    write_hotpotqa_instances_to_filepath(read_hotpotqa_instances(raw_train_path), processed_train_filepath)

    processed_dev_filepath = os.path.join(directory, "dev.jsonl")
    write_hotpotqa_instances_to_filepath(read_hotpotqa_instances(raw_dev_path), processed_dev_filepath)
