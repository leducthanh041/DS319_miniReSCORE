import os
import sqlite3
from typing import Dict, Literal, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class Document:
    id: str
    content: Optional[str] = None
    metadata: Dict[Any, Any] = field(
        default_factory={
            "id": None,
            "title": None,
            "text": None,
        }
    )
    
    def __str__(
        self, 
        max_num_words: int = 350,
        title_prefix: str = "Wikipedia Title: "
    ):
        assert ('title' in self.metadata) and (self.metadata['title'] is not None), (
            f"Document metadata should contain it's title."
        )
        assert ('text' in self.metadata) and (self.metadata['text'] is not None), (
            f"Document metadata should contain it's text."
        )
        
        title, text = self.metadata['title'], self.metadata['text']
        
        text = " ".join(text.split(" ")[:max_num_words]).strip()
        
        document_str = title_prefix + title + "\n" + text
        
        return document_str
    

class Docstore():
    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA busy_timeout=60000")
        conn.commit()

    def __init__(
        self, 
        database_path: str,
        sqlite_file_name: str = 'docstore.db',
        conn: Optional[sqlite3.Connection] = None
        ):
        self.database_path = database_path
        os.makedirs(database_path, exist_ok=True)
        self.sqlite_path = os.path.join(
            database_path, sqlite_file_name
        )
        
        if not conn:
            if os.path.exists(self.sqlite_path):
                raise FileExistsError(
                    f"Database already exists at {self.sqlite_path}. Use `load` to use existing file."
                )
            
            # Initialize DB
            conn = sqlite3.connect(
                self.sqlite_path, 
                check_same_thread=False,
                timeout=60,
            )
            c = conn.cursor()
            c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT
            )
            ''')
            self._configure_connection(conn)
            print(f'Initialize Database at {self.sqlite_path}')
            
        self.conn = conn
        
    
    @classmethod
    def load(
        cls,
        database_path: str,
        sqlite_file_name: str = 'docstore.db'
    ):
        sqlite_path = os.path.join(
            database_path, sqlite_file_name
        )
        conn = sqlite3.connect(
            database=sqlite_path, 
            check_same_thread=False,
            timeout=60,
        )
        cls._configure_connection(conn)
        obj = cls(
            database_path=database_path,
            sqlite_file_name=sqlite_file_name,
            conn=conn
        )
        
        return obj
    
    def add(
        self,
        docs: Union[Document, List[Document]]
    ):
        if type(docs) != list:
            docs = [docs]

        rows = [
            (
                doc.id,
                doc.content,
                json.dumps(doc.metadata),
            )
            for doc in docs
        ]

        c = self.conn.cursor()
        c.executemany(
            '''
            INSERT OR REPLACE INTO documents (id, content, metadata) 
            VALUES (?, ?, ?)
            ''',
            rows,
        )
        self.conn.commit()
        
        
    def _add(
        self, 
        doc: Document
    ):
        # TODO: This can only handle one document per call, It might be more fast to add batch fuction.
        id = doc.id
        content = doc.content
        metadata = json.dumps(doc.metadata) # Convert Dictionary to Json format string.
        
        c = self.conn.cursor()
        c.execute('''
        INSERT OR REPLACE INTO documents (id, content, metadata) 
        VALUES (?, ?, ?)
        ''', (id, content, metadata)
        )
        self.conn.commit()
        
        
    def search(
        self, 
        id: str
    ):
        c = self.conn.cursor()
        c.execute('''
        SELECT id, content, metadata 
        FROM documents 
        WHERE id = ?
        ''', (id,)
        )
        row = c.fetchone()
        
        if row:
            return Document(
                id=row[0], 
                content=row[1], 
                metadata=json.loads(row[2]) if row[2] else None
            )
        else:
            return None
        
    def delete(self, id):
        raise NotImplementedError(
            '...'
        )
