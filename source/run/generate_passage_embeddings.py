import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

from source.module.index.docstore import Document
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.utility import slurm, text_utils


def preprocess_passage_to_doc(passage, args):
    if args.no_title or "title" not in passage:
        text = passage["text"]
    else:
        text = passage["title"] + " " + passage["text"]
    if args.lowercase:
        text = text.lower()
    if args.normalize_text:
        text = text_utils.normalize_text(text)

    return Document(
        id=passage["id"],
        content=text,
        metadata=passage,
    )


def log_loading_progress(bytes_read, file_size, next_report_bytes, report_step_bytes):
    while bytes_read >= next_report_bytes:
        progress = (bytes_read / file_size) * 100 if file_size else 100.0
        print(
            f"Loading passages: {bytes_read / (1024 ** 3):.2f} GiB / "
            f"{file_size / (1024 ** 3):.2f} GiB ({progress:.1f}%).",
            flush=True,
        )
        next_report_bytes += report_step_bytes
    return next_report_bytes


def parse_tsv_passage(line, line_no):
    line = line.rstrip("\n")
    if not line or line == "id\ttext\ttitle":
        return None

    try:
        passage_id, remainder = line.split("\t", 1)
        text, title = remainder.rsplit("\t", 1)
    except ValueError as exc:
        raise ValueError(
            f"Malformed TSV row at line {line_no}: could not parse 3 columns."
        ) from exc

    return {"id": passage_id, "title": title, "text": text}


def iter_sharded_documents(path, shard_id, num_shards, args):
    assert shard_id < num_shards, (
        "Ensure that shard_id is less than num_shards. Note that shard_id always starts from 0."
    )
    assert os.path.exists(path), "Path does't exist."

    file_size = os.path.getsize(path)
    report_step_bytes = max(file_size // 10, 128 * 1024 * 1024) if file_size else 1
    next_report_bytes = report_step_bytes
    bytes_read = 0
    data_row_idx = 0
    shard_doc_count = 0
    malformed_rows = 0

    print(
        f"Streaming shard {shard_id + 1}/{num_shards} from {path}.",
        flush=True,
    )

    with open(path, "r", encoding="utf-8") as fin:
        if path.endswith(".jsonl"):
            for line_no, line in enumerate(fin, start=1):
                bytes_read += len(line.encode("utf-8"))
                next_report_bytes = log_loading_progress(
                    bytes_read, file_size, next_report_bytes, report_step_bytes
                )

                line = line.strip()
                if not line:
                    continue

                if data_row_idx % num_shards != shard_id:
                    data_row_idx += 1
                    continue

                data_row_idx += 1
                shard_doc_count += 1
                yield preprocess_passage_to_doc(json.loads(line), args)
        else:
            for line_no, line in enumerate(fin, start=1):
                bytes_read += len(line.encode("utf-8"))
                next_report_bytes = log_loading_progress(
                    bytes_read, file_size, next_report_bytes, report_step_bytes
                )

                try:
                    passage = parse_tsv_passage(line, line_no)
                except ValueError as exc:
                    malformed_rows += 1
                    if malformed_rows <= 5:
                        print(str(exc), flush=True)
                    continue

                if passage is None:
                    continue

                if data_row_idx % num_shards != shard_id:
                    data_row_idx += 1
                    continue

                data_row_idx += 1
                shard_doc_count += 1
                yield preprocess_passage_to_doc(passage, args)

    if malformed_rows:
        print(
            f"Skipped {malformed_rows} malformed rows while reading {path}.",
            flush=True,
        )
    print(
        f"Finished loading shard {shard_id + 1}/{num_shards}. "
        f"Retained {shard_doc_count} passages for this shard.",
        flush=True,
    )


def cleanup_partial_shard_outputs(output_dir, shard_id):
    shard_prefix = os.path.join(output_dir, f"embeddings_{shard_id:02d}")
    removed_files = 0
    for file_path in glob.glob(f"{shard_prefix}*"):
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed_files += 1

    marker_path = get_shard_marker_path(output_dir, shard_id)
    if os.path.exists(marker_path):
        os.remove(marker_path)

    if removed_files:
        print(
            f"Removed {removed_files} stale embedding chunk files for shard {shard_id:02d}.",
            flush=True,
        )


def get_shard_marker_path(output_dir, shard_id):
    return os.path.join(output_dir, f"embedding_shard_{shard_id:02d}.complete.json")


def embed_passage_batch(retriever, input_texts):
    with torch.no_grad():
        batch_embeddings = retriever._embed_passages(input_texts)
    return batch_embeddings.detach().cpu().numpy()


def save_embedding_chunk(output_dir, shard_id, chunk_index, chunk_embeddings, chunk_documents):
    if not chunk_embeddings:
        return chunk_index

    save_file = os.path.join(
        output_dir,
        f"embeddings_{shard_id:02d}_{chunk_index:04d}",
    )
    embeddings = np.concatenate(chunk_embeddings, axis=0)
    with open(save_file, "wb") as fout:
        pickle.dump((embeddings, chunk_documents), fout, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved chunk {chunk_index:04d} for shard {shard_id:02d}: "
        f"{len(chunk_documents)} passages -> {save_file}",
        flush=True,
    )
    return chunk_index + 1


def log_embedding_progress(total_embedded, batches_processed):
    print(
        f"Embedded {total_embedded} passages across {batches_processed} batches so far.",
        flush=True,
    )


def main(args):
    print(
        f"Load Model, Tokenizer from {args.model_name_or_path}.",
        flush=True,
    )

    cfg = DenseRetrieverConfig(
        batch_size=args.per_gpu_batch_size,
        training_strategy="query_only",
        use_fp16=not args.no_fp16,
        query_model_name_or_path=args.model_name_or_path,
        passage_model_name_or_path=args.model_name_or_path,
        max_length=args.passage_maxlength,
    )
    retriever = DenseRetriever(cfg=cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    cleanup_partial_shard_outputs(args.output_dir, args.shard_id)

    print(
        f"Streaming passages from {args.passages} and saving chunked embeddings to {args.output_dir}.",
        flush=True,
    )

    current_batch_docs = []
    current_batch_texts = []
    chunk_documents = []
    chunk_embeddings = []
    chunk_index = 0
    batches_processed = 0
    total_embedded = 0
    embedding_pbar = tqdm(
        desc="Embedded passages",
        unit="passage",
        mininterval=5.0,
        disable=not sys.stdout.isatty(),
    )

    for document in iter_sharded_documents(args.passages, args.shard_id, args.num_shards, args):
        current_batch_docs.append(document)
        current_batch_texts.append(document.content)

        if len(current_batch_texts) < args.per_gpu_batch_size:
            continue

        batch_embeddings = embed_passage_batch(retriever, current_batch_texts)
        chunk_embeddings.append(batch_embeddings)
        chunk_documents.extend(current_batch_docs)
        total_embedded += len(current_batch_docs)
        batches_processed += 1
        embedding_pbar.update(len(current_batch_docs))

        current_batch_docs = []
        current_batch_texts = []

        if batches_processed % args.progress_report_batches == 0:
            log_embedding_progress(total_embedded, batches_processed)

        if batches_processed % args.save_every_batches == 0:
            chunk_index = save_embedding_chunk(
                args.output_dir,
                args.shard_id,
                chunk_index,
                chunk_embeddings,
                chunk_documents,
            )
            chunk_embeddings = []
            chunk_documents = []
            torch.cuda.empty_cache()

    if current_batch_texts:
        batch_embeddings = embed_passage_batch(retriever, current_batch_texts)
        chunk_embeddings.append(batch_embeddings)
        chunk_documents.extend(current_batch_docs)
        total_embedded += len(current_batch_docs)
        batches_processed += 1
        embedding_pbar.update(len(current_batch_docs))

    chunk_index = save_embedding_chunk(
        args.output_dir,
        args.shard_id,
        chunk_index,
        chunk_embeddings,
        chunk_documents,
    )
    embedding_pbar.close()

    if total_embedded == 0:
        raise ValueError(
            f"No passages were embedded from {args.passages} for shard "
            f"{args.shard_id + 1}/{args.num_shards}."
        )

    marker_path = get_shard_marker_path(args.output_dir, args.shard_id)
    marker_payload = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "passages_embedded": total_embedded,
        "batches_processed": batches_processed,
        "chunk_files_written": chunk_index,
    }
    with open(marker_path, "w", encoding="utf-8") as fout:
        json.dump(marker_payload, fout, indent=2)

    print(
        f"Completed shard {args.shard_id + 1}/{args.num_shards}. "
        f"Embedded {total_embedded} passages into {chunk_index} chunk files. "
        f"Marker: {marker_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to directory containing model weights and config file",
    )
    parser.add_argument(
        "--passages",
        type=str,
        default=None,
        help="Path to passages (.tsv file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="dir path to save embeddings",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Id of the current shard",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=512,
        help="Batch size for the passage encoder forward pass",
    )
    parser.add_argument(
        "--passage_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a passage",
    )
    parser.add_argument(
        "--save_every_batches",
        type=int,
        default=64,
        help="Write one embedding chunk file after this many batches.",
    )
    parser.add_argument(
        "--progress_report_batches",
        type=int,
        default=100,
        help="Print a progress update after this many batches.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="inference in fp32",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="title not added to the passage body",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="lowercase text before encoding",
    )
    parser.add_argument(
        "--normalize_text",
        action="store_true",
        help="lowercase text before encoding",
    )

    args = parser.parse_args()
    slurm.init_distributed_mode(args)
    main(args)
