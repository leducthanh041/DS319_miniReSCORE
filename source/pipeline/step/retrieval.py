import torch
import numpy as np
import random
from tqdm import tqdm
import spacy

from source.utility.data_utils import (
    load_data_from_jsonl
)
from source.utility.system_utils import (
    seed_everything,
)
from source.module.generate.llama import (
    LlamaGenerator,
    LlamaGeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)
from source.pipeline.controller import (
    PipelineController
)
from source.pipeline.state import (
    BaseState,
    QuestionState,
    AnswerState,
    DocumentState,
    ResumeState
)
from source.module.index.docstore import (
    Docstore,
    Document
)
from dataclasses import dataclass
import os
import json

from typing import Optional, Literal, Union, List, Dict, Any


from source.pipeline.utils import (
    parse_path,
    preprocess_documents,
    filter_document,
    preprocess_retrieval_query,
    clean_wrong_json_format
)
from source.pipeline.constants import (
    DOC_DOC_DELIM,
    THOUGHT_THOUGHT_DELIM
)
    
    
class RetrievalStep:
    
    def __init__(
        self,
        cfg,
        retriever,
        indexer,  
        retrieval_trace_file_path: Optional[str] = None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.indexer = indexer
        self.retrieval_trace_file_path = retrieval_trace_file_path

    def _document_to_trace(self, document: Document, rank: int) -> Dict[str, Any]:
        metadata = document.metadata or {}
        return {
            "rank": rank,
            "id": document.id,
            "title": metadata.get("title"),
            "paragraph_text": metadata.get("text") or document.content,
        }

    def _append_retrieval_trace(self, records: List[Dict[str, Any]]) -> None:
        if not self.retrieval_trace_file_path or not records:
            return

        os.makedirs(os.path.dirname(self.retrieval_trace_file_path), exist_ok=True)
        with open(self.retrieval_trace_file_path, "a", encoding="utf-8") as trace_file:
            for record in records:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        
    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        parent_state_ids = []
        question_ids, questions, iteration_numbers = [], [], []
        queries, document_ids_so_far = [], []
        for path in paths:
            parent_state_ids.append(
                path[-1].state_id
            )

            question_id, question, thoughts, documents = parse_path(path)
            question_ids.append(question_id)
            questions.append(question)
            iteration_numbers.append(len(documents) + 1)
            document_ids_so_far.append(
                {doc.id for doc in sum(documents, [])}
            )
            queries.append(
                preprocess_retrieval_query(
                    question,
                    thoughts,
                    retrieval_query_type=self.cfg.retrieval_query_type
                )
            )
            
        embeddings = self.retriever.embed(
            input_texts=queries,
            input_type='query'
        ).detach().cpu().numpy().astype('float32')
        
        indexer_outputs = self.indexer.search(
            query_embeddings=embeddings,
            k=self.cfg.retrieval_buffer_size
        )
        
        all_next_states = []
        trace_records = []
        for (
            indexer_output,
            _document_ids_so_far,
            parent_state_id,
            question_id,
            question,
            query,
            iteration,
        ) in zip(
            indexer_outputs,
            document_ids_so_far,
            parent_state_ids,
            question_ids,
            questions,
            queries,
            iteration_numbers,
        ):
            documents = filter_document(
                documents=indexer_output.documents,
                document_ids_so_far=_document_ids_so_far,
                retrieval_no_duplicates=self.cfg.retrieval_no_duplicates
            )

            all_next_states.append(
                DocumentState(
                    parent_state_id=parent_state_id,
                    documents=documents,
                    question_id=question_id,
                    question=question,
                )
            )

            trace_records.append(
                {
                    "question_id": question_id,
                    "iteration": iteration,
                    "query": query,
                    "retrieval_count": self.cfg.retrieval_count,
                    "retrieval_buffer_size": self.cfg.retrieval_buffer_size,
                    "documents": [
                        self._document_to_trace(document, rank=rank)
                        for rank, document in enumerate(documents, start=1)
                    ],
                }
            )

        self._append_retrieval_trace(trace_records)

        return all_next_states
