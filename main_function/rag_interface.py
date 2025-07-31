"""
Interface module for RAG system integration
"""

import logging
import os
from pathlib import Path
from rag_trainer import ChineseRAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGInterface:
    """A simple interface class for RAG system integration"""
    
    def __init__(self, model_dir=None):
        """Initialize RAG interface with specified model directory"""
        if model_dir is None:
            # Default to 'rag_models' in the same directory
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_models")
        
        self.model_dir = model_dir
        self.rag_system = ChineseRAGSystem(model_save_dir=model_dir)
        self.is_initialized = False
        
    def initialize(self):
        """Load the RAG system"""
        try:
            self.rag_system.load_system()
            self.is_initialized = True
            logger.info("RAG system loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load RAG system: {e}")
            return False
            
    def ask(self, question, api_url=None):
        """Process a question using the RAG system
        
        Args:
            question: The query to process
            api_url: Optional URL for the LLM API endpoint
            
        Returns:
            Answer string from the RAG system
        
        Raises:
            RuntimeError: If the system is not initialized
            ValueError: If the question is empty
        """
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
            
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        # Generate and return answer
        return self.rag_system.generate_answer(question, api_url=api_url)
