# rag_trainer.py

import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict
import torch
import requests

# LlamaIndex imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    logging.warning("FlagEmbeddingReranker not found. Reranker functionality might be limited or require manual import.")
    FlagReranker = None 

# setting the log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseRAGSystem:
    """A class to handle the RAG (Retrieval-Augmented Generation) system for Chinese documents."""
    
    def __init__(self, 
                 processed_texts_dir: str = "processed_texts",
                 model_save_dir: str = "rag_models",
                 embedding_model: str = "qwen/Qwen3-Embedding-0.6B",
                 use_reranker: bool = True,
                 reranker_model: str = "BAAI/bge-multilingual-gemma2"
                 ):
        self.processed_texts_dir = Path(processed_texts_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.embedding_model_name = embedding_model
        self.index = None
        self.retriever = None
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.reranker = None
        self.config = {
                'chunk_size': 512,  
                'chunk_overlap': 100,
                'similarity_top_k': 30,  
                'created_at': datetime.now().isoformat()
            }
    
    def setup_models(self):
        """Setup embedding and reranker models."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Current device: {device}")
        
        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                trust_remote_code=True,
                device=device
            )
            logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Fail to load embedding model: {e}")
            raise
        
        
        Settings.embed_model = self.embedding_model
        
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            separator="。"
        )
        
        # Initialize Reranker if available
        if self.use_reranker and FlagReranker:
            try:
                self.reranker = FlagReranker(
                    self.reranker_model_name
                )
                logger.info(f"Loaded reranker model: {self.reranker_model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.reranker = None
    
    def load_texts(self) -> List[Document]:
        """Load processed text files from the specified directory."""
        documents = []
        if not self.processed_texts_dir.exists():
            logger.error(f"Text directory not found: {self.processed_texts_dir}")
            return documents
        for file_path in self.processed_texts_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    doc = Document(
                        text=content,
                        metadata={
                            'file_name': file_path.name,
                            'source_file': file_path.stem.replace('_processed', ''),
                            'file_path': str(file_path)
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Loaded document: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {e}")
        return documents
    
    def build_index(self, documents: List[Document]):
        """Build vector index"""
        logger.info("Building vector index...")
        try:
            self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
            logger.info("Index built successfully")
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def setup_retriever(self):
        """Setup retriever"""
        if not self.index:
            raise ValueError("Index not built")
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config['similarity_top_k']
        )
        logger.info(f"Retriever setup complete (top_k={self.config['similarity_top_k']})")
    
    def save_system(self):
        """Save RAG system"""
        logger.info("Saving RAG system...")
        index_path = self.model_save_dir / "vector_index"
        if self.index:
            self.index.storage_context.persist(persist_dir=str(index_path))
        config_path = self.model_save_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info("RAG system saved successfully")
    
    def load_system(self):
        """Load RAG system"""
        logger.info("Loading RAG system...")
        config_path = self.model_save_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        self.setup_models()
        index_path = self.model_save_dir / "vector_index"
        if index_path.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            self.index = load_index_from_storage(storage_context)
            self.setup_retriever()
            logger.info("RAG system loaded successfully")
        else:
            logger.error(f"Vector index not found at {index_path}. Please train the system first.")
            raise FileNotFoundError(f"Vector index not found at {index_path}")
    
    def train_system(self, processed_texts_dir: str):
        """Train the RAG system with processed texts."""
        logger.info("Starting RAG system training...")
        self.processed_texts_dir = Path(processed_texts_dir)
        self.setup_models()
        documents = self.load_texts()
        if not documents:
            logger.error("No documents found to train the system.")
            return
        self.build_index(documents)
        self.setup_retriever()
        self.save_system()
        logger.info("RAG system trained successfully")
    
    def generate_answer(self, query: str, api_url: str = "http://localhost:1234/v1/chat/completions") -> str:
        """Generate answer based on user query"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "System not initialized"
        
        # Retrieve documents
        nodes = self.retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found."
        
        # Apply reranking if available
        if self.reranker:
            print("Reranking documents...")
            try:
                node_texts = [node.text for node in nodes]
                pairs = [(query, text) for text in node_texts]
                scores = self.reranker.compute_score(pairs)
                
                # Reorder nodes based on scores
                scored_nodes = list(zip(scores, nodes))
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                nodes = [node for _, node in scored_nodes]
                logger.info(f"Reranking completed, max score: {max(scores):.4f}")
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
        
        # Prepare context
        context_nodes = nodes[:30]  
        context = '\n'.join(
        f"Source: {node.metadata['source_file']}\nContent: {node.text[:500]}"
        for node in context_nodes
    )
        prompt = f"""请根据以下上下文回答问题。如果上下文不包含答案，请说明不知道。无需进行推理或假设。用中文回答全部的问题
上下文：
{context}
问题：{query}
回答："""
        
        payload = {
            "model": "qwen3-14b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30000,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"}, timeout=3000)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            logger.error("API call timed out, please simplify your question or increase timeout")
            return "Thinking too long, please try again later or simplify your question."
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return "Unable to generate answer, please try again later."

# --- Execution Block for RAG Training ---
if __name__ == "__main__":
    PROCESSED_DIR = "processed_texts"
    MODEL_DIR = "rag_models"

    print("\nTraining RAG system...")
    rag_system = ChineseRAGSystem(processed_texts_dir=PROCESSED_DIR, model_save_dir=MODEL_DIR)
    rag_system.train_system(PROCESSED_DIR)
    print("RAG system trained successfully!")