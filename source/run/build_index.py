import os
import argparse
import json
import pickle
import time
import glob
from tqdm import tqdm
import os
from datetime import datetime

import argparse
import pickle

from source.module.index.index import Indexer, IndexerConfig
from source.utility import slurm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def backup_existing_index_artifacts(output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_names = [
        "docstore.db",
        "docstore.db-wal",
        "docstore.db-shm",
        "index.faiss",
        "faiss_id_to_docstore_id.pkl",
    ]
    backed_up = []

    for artifact_name in artifact_names:
        artifact_path = os.path.join(output_dir, artifact_name)
        if not os.path.exists(artifact_path):
            continue

        backup_path = os.path.join(output_dir, f"{artifact_name}.bak.{timestamp}")
        os.replace(artifact_path, backup_path)
        backed_up.append(os.path.basename(backup_path))

    if backed_up:
        print(
            "Backed up existing index artifacts before rebuild: "
            + ", ".join(backed_up),
            flush=True,
        )


def main(args):
    print(
        f"Init Indexer at {args.output_dir}.", flush=True
    )
    backup_existing_index_artifacts(args.output_dir)
    cfg = IndexerConfig(
        embedding_sz=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits,
        database_path=args.output_dir
    )
    indexer = Indexer(
        cfg=cfg
    )
    
    print(
        f"Indexing passages from {args.output_dir}...",
        flush=True,
    )
    pattern = 'embeddings_[0-9][0-9]*'
    matching_files = sorted(glob.glob(
        os.path.join(
            args.output_dir, pattern
        )
    ))
    print(
        f"Found {len(matching_files)} embedding chunk files to index.",
        flush=True,
    )
    if not matching_files:
        raise FileNotFoundError(
            f"No embedding chunk files were found in {args.output_dir}."
        )
    indexed_documents = 0
    pbar = tqdm(
        matching_files, 
        desc="Chunk", 
        postfix={"file": None, "docs": 0}
    )
    for file_path in pbar:
        file_name = os.path.basename(file_path)
        pbar.set_postfix(file=file_name, docs=indexed_documents)
        try:
            with open(file_path, 'rb') as file:
                embeddings, documents = pickle.load(file)
                indexer.index(
                    documents=documents,
                    embeddings=embeddings
                )
                indexed_documents += len(documents)
                pbar.set_postfix(file=file_name, docs=indexed_documents)
                pbar.write(
                    f"Indexed {len(documents)} docs from {file_name}. Total indexed: {indexed_documents}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed while indexing {file_path}") from e
    
    print(
        f"Saving index to {args.output_dir}. Total indexed documents: {indexed_documents}",
        flush=True,
    )
    indexer.save_local(
        override=True
    )
    
    print(
        f"Complete!",
        flush=True,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection_size", type=int, default=768,
        help=""
    )
    parser.add_argument(
        "--n_subquantizers", type=int, default=0,
        help="Number of subquantizer used for embedding quantization, if 0 flat index is used"
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, 
        help="Number of bits per subquantizer"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="dir path to save index"
    )
    args = parser.parse_args()
    slurm.init_distributed_mode(args)
    main(args)
