import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict
import torch
import requests
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LlamaIndex imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage

# Sentence Transformers for custom semantic chunking
from sentence_transformers import SentenceTransformer

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
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 semantic_chunking_model: str = "BAAI/bge-small-zh-v1.5"
                 ):
        self.processed_texts_dir = Path(processed_texts_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.embedding_model_name = embedding_model
        self.semantic_chunking_model = semantic_chunking_model
        self.index = None
        self.retriever = None
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.reranker = None
        self.config = {
            'similarity_top_k': 30,
            'semantic_chunking_threshold': 0.95,  # Cosine similarity threshold for chunking
            'created_at': datetime.now().isoformat()
        }
    
    def setup_models(self, force_offline: bool = False):
        """
        Setup embedding, reranker, and semantic chunking models. 自动检测本地/在线模式。
        # 部署后如需完全离线，请在下方所有 from_pretrained/SentenceTransformer/HuggingFaceEmbedding/FlagReranker 加 local_files_only=True
        """
        import os
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Current device: {device}")

        def is_local_model(model_name):
            # 判断是否为本地路径或cache下有文件
            if os.path.isdir(model_name) or os.path.exists(model_name):
                return True
            # 检查transformers默认cache
            try:
                from transformers.utils.hub import cached_file
                from huggingface_hub.errors import EntryNotFoundError
                try:
                    cached_file(model_name, "config.json", local_files_only=True)
                    return True
                except Exception:
                    return False
            except ImportError:
                return False

        # embedding model
        local_embedding = force_offline or is_local_model(self.embedding_model_name)

        # ====== 这里是 Embedding 模型加载，部署后如需离线请确保 local_files_only=True ======
        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                trust_remote_code=True,
                device=device,
                local_files_only=True # 部署后如需离线请设为 True
            )
            logger.info(f"Successfully loaded embedding model: {self.embedding_model_name} (local_files_only={local_embedding})")
        except Exception as e:
            logger.error(f"Fail to load embedding model: {e}")
            raise

        Settings.embed_model = self.embedding_model

        # semantic chunking model
        local_chunking = force_offline or is_local_model(self.semantic_chunking_model)

        # ====== 这里是 SentenceTransformer 语义分块模型加载，部署后如需离线请确保 local_files_only=True ======
        try:
            self.semantic_chunker = SentenceTransformer(
                self.semantic_chunking_model,
                device=device,
                local_files_only= True # 部署后如需离线请设为 True
            )
            logger.info(f"Initialized semantic chunking model: {self.semantic_chunking_model} (local_files_only={local_chunking})")
        except Exception as e:
            logger.error(f"Failed to initialize semantic chunking model: {e}")
            raise

        # reranker
        # ====== 这里是 FlagReranker 重排序模型加载，部署后如需离线请确保 local_files_only=True ======
        if self.use_reranker and FlagReranker:
            local_reranker = force_offline or is_local_model(self.reranker_model_name)
            try:
                try:
                    self.reranker = FlagReranker(
                        self.reranker_model_name,
                        local_files_only=True # 部署后如需离线请设为 True
                    )
                except TypeError:
                    self.reranker = FlagReranker(
                        self.reranker_model_name
                    )
                logger.info(f"Loaded reranker model: {self.reranker_model_name} (local_files_only={local_reranker})")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.reranker = None
    
    def custom_semantic_chunking(self, text: str) -> List[str]:
        """Custom semantic chunking using Sentence Transformers."""
        try:
            # Split text into sentences using Chinese punctuation
            sentences = re.split(r'(?<=[。！？])', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                logger.warning("No valid sentences found for chunking")
                return [text]
            
            # Embed sentences
            embeddings = self.semantic_chunker.encode(sentences, convert_to_numpy=True)
            
            # Group sentences into chunks based on cosine similarity
            chunks = []
            current_chunk = [sentences[0]]
            current_embeddings = [embeddings[0]]
            
            for i in range(1, len(sentences)):
                # Compute similarity between current sentence and last sentence in chunk
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    current_embeddings[-1].reshape(1, -1)
                )[0][0]
                
                if sim >= self.config['semantic_chunking_threshold']:
                    # Add to current chunk if similar
                    current_chunk.append(sentences[i])
                    current_embeddings.append(embeddings[i])
                else:
                    # Start new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_embeddings = [embeddings[i]]
            
            # Add the last chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Filter out very short chunks
            chunks = [chunk for chunk in chunks if len(chunk) > 10]
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            return [text]  # Fallback to original text
    
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
        """Build vector index with custom semantic chunking."""
        logger.info("Building vector index with semantic chunking...")
        try:
            # Convert LlamaIndex documents to semantic chunks
            chunked_documents = []
            for doc in documents:
                # Perform custom semantic chunking
                chunks = self.custom_semantic_chunking(doc.text)
                for i, chunk in enumerate(chunks):
                    # Create new LlamaIndex Document for each chunk
                    chunk_doc = Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': f"{doc.metadata['file_name']}_chunk_{i}",
                            'chunk_index': i
                        }
                    )
                    chunked_documents.append(chunk_doc)
                logger.info(f"Chunked document {doc.metadata['file_name']} into {len(chunks)} chunks")
            
            # Build index with chunked documents
            self.index = VectorStoreIndex.from_documents(chunked_documents, show_progress=True)
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
    
    def retrieve_relevant_docs(self, query: str) -> str:
        """Retrieve relevant documents and return formatted context"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "System not initialized"
        
        # Retrieve documents
        nodes = self.retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found."
        
        # Apply reranking if available
        if self.reranker:
            logger.info("Reranking documents...")
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
        
        return context
    
    def generate_answer(self, query: str, api_url: str = "http://localhost:1234/v1/chat/completions") -> str:
        """Generate answer based on user query using consistent model name"""
        context = self.retrieve_relevant_docs(query)
        
        if context in ["System not initialized", "No relevant documents found."]:
            return context
        
        prompt = f"""请根据以下上下文回答问题。如果上下文不包含答案，请说明不知道。用中文或者英文回答全部的问题。
在回答之前，请在 <think> 标签中提供你的推理。

上下文：
{context}

问题：{query}

回答："""
        
        # Use the same model name as other parts of the system
        payload = {
            "model": "qwen/qwen3-14b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30000,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"}, timeout=300)
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
