from pathlib import Path
import torch
from typing import List, Any,Optional
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import sys
from dotenv import load_dotenv
import logging
from transformers import AutoModel, AutoTokenizer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
class Qwen3VLEmbedding(BaseEmbedding):
    """Custom Embedding class for Qwen3-VL models."""
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(self, model_name: str, device: str = "cuda", cache_folder: str = "", **kwargs):
        """
        Initialize the Qwen3VLEmbedding class.
        Args:
            model_name (str): The name or path of the pre-trained Qwen3-VL model.
            device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
            cache_folder (str): The folder to cache the model and tokenizer.
        """
        super().__init__(model_name=model_name, **kwargs)
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_folder, 
            trust_remote_code=True
        )
        
        self._model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=cache_folder, 
            trust_remote_code=True,
            device_map=device, 
            output_hidden_states=True
        )
        self._model.eval()

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get the embedding for a query string.
        Args:
            query (str): The input query string.
        Returns:
            List[float]: The embedding vector for the query.
        """
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a text string.
        Args:
            text (str): The input text string.
        Returns:
            List[float]: The embedding vector for the text.
        """
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            # use mean pooling on the last hidden state
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            return embedding[0].cpu().numpy().tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Async method to get the embedding for a query string.
        Args:
            query (str): The input query string.
        Returns:
            List[float]: The embedding vector for the query.
        """
        return self._get_query_embedding(query)

class Qwen3VLReranker:
    """Custom Reranker class for Qwen3-VL models acting as CrossEncoder."""
    
    def __init__(self, model_name: str, device: str = "cuda", cache_folder: str = ""):
        """
        Initialize the Qwen3VLReranker class.
        Args:
            model_name (str): The name or path of the pre-trained Qwen3-VL model.
            device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
            cache_folder (str): The folder to cache the model and tokenizer.
        """
        self.device = device
        # Use Qwen3VLEmbedding logic to treat it as a Bi-Encoder for stability
        # or we could use generation. Here we use similarity of embeddings 
        # because Qwen3-VL is not trained as a classifier (CrossEncoder).
        self.embedding_model = Qwen3VLEmbedding(model_name, device=device, cache_folder=cache_folder)

    def predict(self, sentences: List[List[str]]) -> List[float]:
        """
        Predict similarity scores for a list of (query, document) pairs.
        Args:
            sentences: A list of [query, document] pairs.
        Returns:
            A list of float scores.
        """
        scores = []
        # Process pairs
        for query, doc in sentences:
            # We compute embeddings for query and doc separately
            q_emb = torch.tensor(self.embedding_model._get_query_embedding(query)).to(self.device)
            d_emb = torch.tensor(self.embedding_model._get_text_embedding(doc)).to(self.device)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0))
            scores.append(similarity.item())
        return scores

class RAGSystem:
    """
    A class to handle the Retrieval-Augmented Generation (RAG) system.
    This class manages the embedding model, reranker, semantic chunking,
    and other components of the RAG pipeline.
    """

    # Initialize the RAG system
    def __init__(self, 
                processed_texts_dir: str = "processed_texts",
                model_save_dir: str = "rag_models",
                embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                use_reranker: bool = True,
                reranker_model: str = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B"),
                semantic_chunking_model: str = os.getenv("SEMANTIC_CHUNKING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                shared_embed_model = None,
                shared_reranker = None,
                shared_chunker = None):
        """Initialize the RAG system."""
        self.processed_texts_dir = Path(processed_texts_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.embedding_model_name = embedding_model
        self.semantic_chunking_model = semantic_chunking_model
        self.index = None
        self.retriever = None
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        
        # Store shared instances
        self.embedding_model: Optional[BaseEmbedding] = shared_embed_model
        self.reranker: Optional[Any] = shared_reranker
        self.semantic_chunker: Optional[SentenceTransformer] = shared_chunker
        
        self.config = {
            'similarity_top_k': 30,
            'semantic_chunking_threshold': 0.7,
        }

    # Setup models for RAG system
    def setup_models(self, force_offline: bool = True):
        """
        Setup embedding, reranker, and semantic chunking models.
        Args:
            force_offline (bool): If True, forces loading models from local cache only.
        """
        
        # If models are already injected, skip loading
        if self.embedding_model and self.semantic_chunker:
            Settings.embed_model = self.embedding_model
            if self.use_reranker and not self.reranker:
                 logger.warning("Reranker requested but not provided in shared instances. It will be loaded individually.")
            else:
                 return # Models are ready

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine local model directory
        if getattr(sys, 'frozen', False):
            # If running as packaged executable
            application_path = Path(sys.executable).parent
            local_model_dir = application_path / "model"
        else:
            # If running in development environment
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            local_model_dir = project_root / "model"
        
        try:
            # Debug: Log the determined model directory path
            logger.info(f"Attempting to load models from: {local_model_dir}")
            logger.info(f"Model directory exists: {local_model_dir.exists()}")
            if local_model_dir.exists():
                logger.info(f"Contents of model directory: {list(local_model_dir.iterdir())}")
            
            # Set environment variables to use local model directory
            os.environ['HF_HOME'] = str(local_model_dir)
            os.environ['TRANSFORMERS_CACHE'] = str(local_model_dir)
            os.environ['HF_DATASETS_CACHE'] = str(local_model_dir)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(local_model_dir)
            
            # Try to load embedding model from local directory first
            # Use the model name from initialization
            embedding_model_name = self.embedding_model_name
            
            # Check if using Qwen3-VL for embedding
            if "Qwen3-VL" in embedding_model_name:
                logger.info(f"Detected Qwen3-VL model for embedding. Using custom Qwen3VLEmbedding class.")
                self.embedding_model = Qwen3VLEmbedding(
                    model_name=embedding_model_name,
                    device=device,
                    cache_folder=str(local_model_dir)
                )
            else:
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=embedding_model_name,
                    device=device,
                    cache_folder=str(local_model_dir)
                )
                
            Settings.embed_model = self.embedding_model
            logger.info(f"Successfully loaded embedding model from local cache: {local_model_dir}")

            # Load semantic chunking model from local directory
            self.semantic_chunker = SentenceTransformer(
                self.semantic_chunking_model,
                device=device,
                cache_folder=str(local_model_dir)
            )
            logger.info(f"Successfully loaded semantic chunking model from local cache: {local_model_dir}")

            # Load reranker model from local directory
            if self.use_reranker:
                # Check if using Qwen3-VL for reranker
                if "Qwen3-VL" in self.reranker_model_name:
                    logger.info(f"Detected Qwen3-VL model for reranker. Using custom Qwen3VLReranker class.")
                    self.reranker = Qwen3VLReranker(
                        model_name=self.reranker_model_name,
                        device=device,
                        cache_folder=str(local_model_dir)
                    )
                else:
                    self.reranker = CrossEncoder(
                        self.reranker_model_name,
                        device=device,
                        cache_folder=str(local_model_dir)
                    )
                logger.info(f"Successfully loaded reranker model from local cache: {local_model_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to load models from local cache ({local_model_dir}): {e}")
            logger.info("Attempting to load models from online/default cache...")
            
            # Fallback: Load models from online or default cache
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

    def custom_semantic_chunking(self, text: str) -> List[str]:
        """
        Custom semantic chunking using Sentence Transformers.
        Args:
            text (str): The input text to be chunked.
        Returns:
            List[str]: A list of semantically chunked text segments.
        """
        if self.semantic_chunker is None:
            raise ValueError("Semantic chunker model is not initialized.")
        
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
        """
        Load processed text files from the specified directory.
        Returns:
            List[Document]: A list of Document objects with text and metadata.
        """
        
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

    def build_index(self, documents: List[Document]):
        """
        Build vector index with custom semantic chunking.
        Args:
            documents (List[Document]): List of Document objects to index.
        """
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

    def setup_retriever(self):
        """Setup retriever"""
        if not self.index:
            raise ValueError("Index not built")
        
        if isinstance(self.index, VectorStoreIndex):
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
            self.write_debug_to_file(query, context_nodes, [], "retrieval_debug.txt")
        
        context = '\n'.join(
            f"Source: {node.metadata['source_file']}\nContent: {node.text[:10000]}"
            for node in context_nodes
        )
        return context

    def write_debug_to_file(self, query: str, context_nodes: list, scores: list = [], filename: str = "debug_output.txt"):
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
                
