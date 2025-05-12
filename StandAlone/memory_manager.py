import os
import pickle
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Lock, Thread
import logging
from datetime import datetime
from queue import Queue, Empty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-load models at module level for better performance
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = EMBEDDER.get_sentence_embedding_dimension()

class MemoryManager:
    def __init__(self, 
                 max_tokens=2048,
                 memory_limit=1000,
                 index_path="faiss.index",
                 memory_path="memory.pkl"):
        
        self.max_tokens = max_tokens
        self.memory_limit = memory_limit
        self.recent_history: List[Dict] = []
        self.long_term_memory: List[Dict] = []
        self.vector_map = []
        
        self.index_path = index_path
        self.memory_path = memory_path
        self.lock = Lock()
        self.last_save = datetime.now()
        self.save_queue = Queue()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        
        # Start background saver thread
        self.saver_thread = Thread(target=self._background_saver, daemon=True)
        self.saver_thread.start()
        
        # Async load existing data
        self._load_memory_async()

    def _background_saver(self):
        """Dedicated thread for handling save operations"""
        while True:
            try:
                # Wait for save commands with timeout
                save_type = self.save_queue.get(timeout=60.0)
                
                if save_type == "full":
                    self._safe_save()
                elif save_type == "index":
                    self._save_index_only()
                    
            except Empty:
                # Periodic flush even if no requests
                if (datetime.now() - self.last_save).total_seconds() > 300:  # 5 min
                    self._safe_save()
            except Exception as e:
                logger.error(f"Background saver error: {str(e)}")

    def _safe_save(self):
        """Thread-safe full save operation"""
        try:
            with self.lock:
                memory_copy = self.long_term_memory.copy()
                vector_copy = self.vector_map.copy()
                index_copy = faiss.clone_index(self.index)
            
            # Save memory data
            temp_path = self.memory_path + ".tmp"
            with open(temp_path, "wb") as f:
                pickle.dump({
                    "long_term_memory": memory_copy,
                    "vector_map": vector_copy
                }, f)
            os.replace(temp_path, self.memory_path)
            
            # Save FAISS index
            temp_idx = self.index_path + ".tmp"
            faiss.write_index(index_copy, temp_idx)
            os.replace(temp_idx, self.index_path)
            
            self.last_save = datetime.now()
            logger.debug("Full memory state saved")
            
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")

    def _save_index_only(self):
        """Quick save of just the FAISS index"""
        try:
            with self.lock:
                index_copy = faiss.clone_index(self.index)
            
            temp_idx = self.index_path + ".tmp"
            faiss.write_index(index_copy, temp_idx)
            os.replace(temp_idx, self.index_path)
            
            logger.debug("Index saved")
        except Exception as e:
            logger.error(f"Index save failed: {str(e)}")

    def _load_memory_async(self):
        """Async memory loading"""
        try:
            if os.path.exists(self.index_path):
                with self.lock:
                    self.index = faiss.read_index(self.index_path)
                
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "rb") as f:
                    data = pickle.load(f)
                    with self.lock:
                        self.long_term_memory = data.get("long_term_memory", [])
                        self.vector_map = data.get("vector_map", [])
                        
        except Exception as e:
            logger.error(f"Memory load failed: {str(e)}")

    def add_message(self, role: str, content: str):
        """Optimized message addition with async processing"""
        start_time = datetime.now()
        content = content[:5000]  # Truncate very long messages
        
        try:
            # Generate embedding first (most time-consuming part)
            embedding = EMBEDDER.encode([content])[0]
            
            with self.lock:
                message = {"role": role, "content": content}
                self.recent_history.append(message)
                self.long_term_memory.append(message)
                
                # Add to FAISS index
                self.index.add(np.array([embedding]))
                self.vector_map.append(len(self.long_term_memory) - 1)
                
                self._trim_history()
                
            # Trigger background save if needed
            if (datetime.now() - self.last_save).total_seconds() > 60:
                self.save_queue.put("full")
            else:
                self.save_queue.put("index")  # Just save the index
                
        except Exception as e:
            logger.error(f"Add message failed: {str(e)}")
            raise

        duration = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Added message in {duration:.3f}s")

    def _trim_history(self):
        """In-memory trimming optimized for performance"""
        # Token-based trimming
        total_chars = sum(len(m["content"]) for m in self.recent_history)
        while total_chars > self.max_tokens * 4 and len(self.recent_history) > 1:
            self.recent_history.pop(0)
            total_chars = sum(len(m["content"]) for m in self.recent_history)
            
        # Absolute limit enforcement
        if len(self.long_term_memory) > self.memory_limit:
            excess = len(self.long_term_memory) - self.memory_limit
            self.long_term_memory = self.long_term_memory[excess:]
            self.vector_map = self.vector_map[excess:]
            self.rebuild_faiss_index()

    def rebuild_faiss_index(self):
        """Optimized index rebuilding"""
        try:
            new_index = faiss.IndexFlatL2(EMBED_DIM)
            embeddings = []
            
            for msg in self.long_term_memory:
                embeddings.append(EMBEDDER.encode([msg["content"]])[0])
                
            if embeddings:
                new_index.add(np.array(embeddings))
                
            with self.lock:
                self.index = new_index
                self.vector_map = list(range(len(self.long_term_memory)))
                
        except Exception as e:
            logger.error(f"Index rebuild failed: {str(e)}")
            raise

    def retrieve_relevant(self, query: str, top_k=3) -> List[Dict]:
        """Efficient retrieval with error handling"""
        if not self.long_term_memory:
            return []
            
        try:
            query_vec = EMBEDDER.encode([query[:1000]])[0]  # Truncate long queries
            
            with self.lock:
                D, I = self.index.search(np.array([query_vec]), top_k)
                return [self.long_term_memory[self.vector_map[i]] 
                       for i in I[0] if i < len(self.vector_map)]
                       
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    def summarize_old(self) -> str:
        """Fast fallback summarization"""
        with self.lock:
            old_memory = self.long_term_memory[:-len(self.recent_history)] if self.recent_history else self.long_term_memory
            
            if not old_memory:
                return "No previous context"
                
            combined = " ".join([m["content"] for m in old_memory])
            return f"Previous context: {combined[:500]}..."  # Simple truncation

    def get_context(self, query: str) -> List[Dict]:
        """Safe context building with fallbacks"""
        try:
            summary = self.summarize_old()
            relevant = self.retrieve_relevant(query) or []
            return [{"role": "system", "content": summary}] + relevant + self.recent_history
        except Exception as e:
            logger.error(f"Context build failed: {str(e)}")
            return self.recent_history[-10:]  # Fallback to recent messages

    def __del__(self):
        """Cleanup on destruction"""
        self._safe_save()
