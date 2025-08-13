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
                 semantic_chunking_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
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
    
    def setup_models(self, force_offline: bool = True):
        """
        Setup embedding, reranker, and semantic chunking models. 默认使用离线模式。
        """
        import os
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Current device: {device}")

        # 强制设置离线模式环境变量
        os.environ['HF_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        def create_sentence_transformers_config(model_path):
            """为缺失 sentence_transformers_config.json 的模型创建基础配置"""
            config_file = os.path.join(model_path, "sentence_transformers_config.json")
            if not os.path.exists(config_file):
                # 尝试读取模型的实际配置来获取正确的维度
                try:
                    model_config_file = os.path.join(model_path, "config.json")
                    if os.path.exists(model_config_file):
                        with open(model_config_file, 'r', encoding='utf-8') as f:
                            model_config = json.load(f)
                        
                        # 尝试获取嵌入维度
                        embedding_dim = model_config.get('hidden_size', 
                                       model_config.get('model_dim', 
                                       model_config.get('d_model', 1024)))
                    else:
                        embedding_dim = 1024  # 默认维度
                except Exception:
                    embedding_dim = 1024  # 默认维度
                
                # 基础的 sentence-transformers 配置
                config = {
                    "__version__": {
                        "sentence_transformers": "2.2.2",
                        "transformers": "4.21.0",
                        "pytorch": "1.12.1"
                    },
                    "modules": [
                        {
                            "idx": 0,
                            "name": "0",
                            "path": "",
                            "type": "sentence_transformers.models.Transformer"
                        },
                        {
                            "idx": 1,
                            "name": "1",
                            "path": "1_Pooling",
                            "type": "sentence_transformers.models.Pooling"
                        }
                    ]
                }
                
                try:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                    
                    # 创建 pooling 配置目录和文件
                    pooling_dir = os.path.join(model_path, "1_Pooling")
                    os.makedirs(pooling_dir, exist_ok=True)
                    
                    pooling_config = {
                        "word_embedding_dimension": embedding_dim,
                        "pooling_mode_cls_token": False,
                        "pooling_mode_mean_tokens": True,
                        "pooling_mode_max_tokens": False,
                        "pooling_mode_mean_sqrt_len_tokens": False
                    }
                    
                    pooling_config_file = os.path.join(pooling_dir, "config.json")
                    with open(pooling_config_file, 'w', encoding='utf-8') as f:
                        json.dump(pooling_config, f, indent=2)
                    
                    logger.info(f"Created sentence_transformers_config.json for {model_path} with embedding_dim={embedding_dim}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to create config for {model_path}: {e}")
                    return False
            return True

        def find_local_model_path(model_name):
            """查找本地模型路径"""
            # 检查是否是绝对路径
            if os.path.isabs(model_name) and os.path.exists(model_name):
                return model_name
                
            # 检查当前缓存目录
            cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', ''))
            if cache_dir:
                # 尝试标准的 HuggingFace 模型路径格式
                model_cache_name = model_name.replace('/', '--')
                model_dirs = [
                    os.path.join(cache_dir, f"models--{model_cache_name}"),
                    os.path.join(cache_dir, model_cache_name),
                    os.path.join(cache_dir, model_name)
                ]
                
                for model_dir in model_dirs:
                    if os.path.exists(model_dir):
                        # 查找 snapshots 目录下的实际模型
                        snapshots_dir = os.path.join(model_dir, "snapshots")
                        if os.path.exists(snapshots_dir):
                            # 获取最新的快照目录
                            snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                           if os.path.isdir(os.path.join(snapshots_dir, d))]
                            if snapshot_dirs:
                                latest_snapshot = sorted(snapshot_dirs)[-1]
                                full_path = os.path.join(snapshots_dir, latest_snapshot)
                                if os.path.exists(os.path.join(full_path, "config.json")):
                                    logger.info(f"Found local model at: {full_path}")
                                    return full_path
                        
                        # 直接检查是否有 config.json
                        if os.path.exists(os.path.join(model_dir, "config.json")):
                            logger.info(f"Found local model at: {model_dir}")
                            return model_dir
            
            # 如果都找不到，返回原始名称（可能会失败）
            logger.warning(f"Could not find local model for: {model_name}")
            return model_name

        # ====== Embedding 模型加载 ======
        try:
            embedding_model_path = find_local_model_path(self.embedding_model_name)
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_model_path,
                trust_remote_code=True,
                device=device,
                local_files_only=True
            )
            logger.info(f"Successfully loaded embedding model: {embedding_model_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        Settings.embed_model = self.embedding_model

        # ====== SentenceTransformer 语义分块模型加载 ======
        try:
            chunking_model_path = find_local_model_path(self.semantic_chunking_model)
            
            # 尝试创建缺失的 sentence-transformers 配置
            create_sentence_transformers_config(chunking_model_path)
            
            # 使用更强的警告抑制
            import warnings
            import logging
            
            # 临时降低 sentence_transformers 的日志级别
            st_logger = logging.getLogger('sentence_transformers')
            original_level = st_logger.level
            st_logger.setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 抑制所有相关警告
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*No sentence-transformers model found.*")
                warnings.filterwarnings("ignore", message=".*Creating a new one with mean pooling.*")
                
                self.semantic_chunker = SentenceTransformer(
                    chunking_model_path,
                    device=device,
                    local_files_only=True,
                    trust_remote_code=True
                )
            
            # 恢复原始日志级别
            st_logger.setLevel(original_level)
            
            logger.info(f"Initialized semantic chunking model: {chunking_model_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize semantic chunking model: {e}")
            raise

        # ====== FlagReranker 重排序模型加载 ======
        if self.use_reranker and FlagReranker:
            try:
                reranker_model_path = find_local_model_path(self.reranker_model_name)
                try:
                    self.reranker = FlagReranker(
                        reranker_model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except TypeError:
                    # 如果参数不支持，则使用基本参数
                    self.reranker = FlagReranker(
                        reranker_model_path,
                        trust_remote_code=True
                    )
                logger.info(f"Loaded reranker model: {reranker_model_path}")
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

