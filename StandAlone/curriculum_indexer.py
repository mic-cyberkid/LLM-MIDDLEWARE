# curriculum_indexer.py

import os
import pdfplumber
import numpy as np
import json
import faiss
import time
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


class FaissCache:
    def __init__(self, expiry_seconds=300):
        self.cache = {}
        self.expiry = expiry_seconds

    def get(self, query):
        item = self.cache.get(query)
        if item:
            ts, result = item
            if time.time() - ts < self.expiry:
                return result
            else:
                del self.cache[query]
        return None

    def set(self, query, result):
        self.cache[query] = (time.time(), result)


class CurriculumIndexer:
    def __init__(
        self,
        pdf_folder: str,
        index_file: str = "data/pdf_faiss.index",
        metadata_file: str = "data/pdf_metadata.json",
        chunk_size: int = 200
    ):
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.chunk_size = chunk_size
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = FaissCache()
        self.index = None
        self.metadata = []

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self._load_index()
        else:
            self.build_index()


    def _extract_text_from_pdf(self,pdf_path):
        texts = " "
        if pdf_path.lower().endswith(".pdf"):
            print("working on ",pdf_path)
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(filter(None, (page.extract_text() for page in pdf.pages)))
                texts += text.strip()
        return texts

    def _chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks, current, length = [], [], 0
        for s in sentences:
            s_len = len(s.split())
            if length + s_len > self.chunk_size:
                chunks.append(" ".join(current))
                current, length = [s], s_len
            else:
                current.append(s)
                length += s_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _get_embedding(self, text):
        return self.embed_model.encode(text)

    def build_index(self):
        embeddings = []
        metadata = []

        for file in os.listdir(self.pdf_folder):
            if not file.endswith(".pdf"):
                continue
            path = os.path.join(self.pdf_folder, file)
            text = self._extract_text_from_pdf(path)
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                emb = self._get_embedding(chunk)
                embeddings.append(emb)
                metadata.append({"file": file, "chunk_idx": i, "text": chunk})

        if not embeddings:
            raise ValueError("No PDF content processed for indexing.")

        data_np = np.array(embeddings).astype('float32')
        dim = data_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(data_np)
        self.metadata = metadata

        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def _load_index(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)

    def search(self, query: str, top_k: int = 5):
        cached = self.cache.get(query)
        if cached:
            return cached

        query_vec = np.array([self._get_embedding(query)]).astype('float32')
        D, I = self.index.search(query_vec, top_k)
        results = [self.metadata[i]['text'] for i in I[0] if i < len(self.metadata)]

        self.cache.set(query, results)
        return results
