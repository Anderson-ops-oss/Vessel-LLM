from fileinput import filename
from pathlib import Path
import torch
from typing import List
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Debug
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Chinese RAG System
class ChineseRAGSystem:
    """A class to handle the RAG system for Chinese and English documents."""

    # Initialize the RAG system
    def __init__(self, 
                 processed_texts_dir: str = "processed_texts",
                 model_save_dir: str = "rag_models",
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 use_reranker: bool = True,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 semantic_chunking_model: str = "BAAI/bge-m3"):
        """Initialize the RAG system.
        Args:
            processed_texts_dir: Directory containing processed text documents
            model_save_dir: Directory to save trained models
            embedding_model: Name of the embedding model to use
            use_reranker: Whether to use a reranker model
            reranker_model: Name of the reranker model to use
            semantic_chunking_model: Name of the semantic chunking model to use
        """
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
            'semantic_chunking_threshold': 0.7,
        }

    # Setup models for RAG system
    def setup_models(self, force_offline: bool = True) -> None:
        """Setup embedding, reranker, and semantic chunking models."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if force_offline:
            try:
                os.environ['HF_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
            
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=self.embedding_model_name,
                    device=device,
                    local_files_only=force_offline
                )
                Settings.embed_model = self.embedding_model

                # Load semantic chunking model
                self.semantic_chunker = SentenceTransformer(
                    self.semantic_chunking_model,
                    device=device,
                    local_files_only=force_offline
                )

                # Load reranker model
                self.reranker = CrossEncoder(
                    self.reranker_model_name,
                    device=device,
                    local_files_only=force_offline
                )
                
            except Exception as e:
            # Load embedding model
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=self.embedding_model_name,
                    device=device,
                    local_files_only=False
                )
                Settings.embed_model = self.embedding_model

                # Load semantic chunking model
                self.semantic_chunker = SentenceTransformer(
                    self.semantic_chunking_model,
                    device=device,
                    local_files_only=False
                )

                # Load reranker model
                if self.use_reranker:
                    self.reranker = CrossEncoder(
                        self.reranker_model_name,
                        device=device,
                        local_files_only=False
                    )

    # Custom semantic chunking
    def custom_semantic_chunking(self, text: str) -> List[str]:
        """Custom semantic chunking using Sentence Transformers."""
        # Split text into sentences
        sentences = re.split(r'(?<=[。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]
        
        # Encode sentences
        embeddings = self.semantic_chunker.encode(sentences, convert_to_numpy=True)
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                current_embeddings[-1].reshape(1, -1)
            )[0][0]
            
            if sim >= self.config['semantic_chunking_threshold']:
                current_chunk.append(sentences[i])
                current_embeddings.append(embeddings[i])
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embeddings = [embeddings[i]]
        
        if current_chunk:
            # print(current_chunk[:3])
            chunks.append(' '.join(current_chunk))

        # Only keep chunks longer than 10 characters
        chunks = [chunk for chunk in chunks if len(chunk) > 10]
        return chunks
        

    def load_texts(self) -> List[Document]:
        """Load processed text files from the specified directory."""
        
        documents = []
        if not self.processed_texts_dir.exists():
            return documents
        for file_path in self.processed_texts_dir.glob("*.txt"):
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
        return documents

    # Build vector index
    def build_index(self, documents: List[Document]) -> None:
        """Build vector index with custom semantic chunking."""
        chunked_documents = []
        for doc in documents:
            chunks = self.custom_semantic_chunking(doc.text)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': f"{doc.metadata['file_name']}_chunk_{i}",
                        'chunk_index': i
                    }
                )
                chunked_documents.append(chunk_doc)
        self.index = VectorStoreIndex.from_documents(chunked_documents)

    # Setup retriever
    def setup_retriever(self) -> None:
        """Setup retriever"""
        if not self.index:
            raise ValueError("Index not built")
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config['similarity_top_k']
        )
    
    # Save RAG system
    def save_system(self) -> None:
        """Save RAG system"""
        index_path = self.model_save_dir / "vector_index"
        if self.index:
            self.index.storage_context.persist(persist_dir=str(index_path))
        config_path = self.model_save_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    # Load RAG system
    def load_system(self) -> None:
        """Load RAG system"""
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
        else:
            raise FileNotFoundError(f"Vector index not found at {index_path}")
    
    # Train the RAG system
    def train_system(self, processed_texts_dir: str) -> None:
        """Train the RAG system with processed texts."""
        self.processed_texts_dir = Path(processed_texts_dir)
        self.setup_models()
        documents = self.load_texts()
        if not documents:
            return
        self.build_index(documents)
        self.setup_retriever()
        self.save_system()

    def retrieve_relevant_docs(self, query: str) -> str:
        """Retrieve relevant documents and return formatted context"""
        if not self.retriever:
            return "System not initialized"
        
        nodes = self.retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found."

        if self.reranker:
            node_texts = [node.text for node in nodes]
            pairs = [[query, text] for text in node_texts]
            scores = self.reranker.predict(pairs)
            scored_nodes = list(zip(scores, nodes))
            
            # Sort by score (highest first)
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            nodes = [node for _, node in scored_nodes]
            
            # Get top 10 and their scores
            context_nodes = nodes[:5]
            top_scores = [score for score, _ in scored_nodes[:5]]
            
            # Write debug info to file
            self.write_debug_to_file(query, context_nodes, top_scores, "reranker_debug.txt")
            
        else:
            context_nodes = nodes[:5]
            # Write debug info to file
            self.write_debug_to_file(query, context_nodes, None, "retrieval_debug.txt")
        
        context = '\n'.join(
            f"Source: {node.metadata['source_file']}\nContent: {node.text[:10000]}"
            for node in context_nodes
        )
        return context

    def write_debug_to_file(self, query: str, context_nodes: list, scores: list = None, filename: str = "debug_output.txt"):
        """Write debug information to a text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write("="*50 + "\n")
            
            if scores:
                f.write("Ranked Results (with reranker scores):\n")
                for i, (node, score) in enumerate(zip(context_nodes, scores)):
                    f.write(f"\nRank {i+1} - Score: {score:.4f}\n")
                    f.write(f"Source: {node.metadata.get('source_file', 'Unknown')}\n")
                    f.write(f"Chunk ID: {node.metadata.get('chunk_id', 'Unknown')}\n")
                    f.write(f"Content: {node.text}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("Results (no reranker):\n")
                for i, node in enumerate(context_nodes):
                    f.write(f"\nRank {i+1}\n")
                    f.write(f"Source: {node.metadata.get('source_file', 'Unknown')}\n")
                    f.write(f"Chunk ID: {node.metadata.get('chunk_id', 'Unknown')}\n")
                    f.write(f"Content: {node.text}\n")
                    f.write("-" * 30 + "\n")
                
