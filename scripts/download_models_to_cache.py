import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Downloading embedding model
def download_embedding_model(model_name: str, cache_dir: str):
    print(f"Downloading embedding model: {model_name} ...")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Embedding model downloaded to {cache_dir}")


# Downloading reranker model
def download_reranker_model(model_name: str, cache_dir: str):
    print(f"Downloading reranker model: {model_name} ...")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Reranker model downloaded to {cache_dir}")


# Downloading semantic chunking model
def download_semantic_chunking_model(model_name: str, cache_dir: str):
    print(f"Downloading semantic chunking model: {model_name} ...")
    SentenceTransformer(model_name, cache_folder=cache_dir)
    print(f"Semantic chunking model downloaded to {cache_dir}")

# Downloading conversation model
def download_conversation_model(model_name: str, cache_dir: str):
    print(f"Downloading conversation model: {model_name} ...")
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Conversation model downloaded to {cache_dir}")


if __name__ == "__main__":
    # Defining model names
    EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B" 
    RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B" 
    SEMANTIC_CHUNKING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
    CONVERSACTION_MODEL = "Qwen/Qwen3-VL-8B-Thinking"


    # Setting the cache folder path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {BASE_DIR}")
    CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'model'))
    print(f"Cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Setting environment variables to ensure transformers download to the specified cache
    os.environ['HF_HOME'] = CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR

    # Downloading models
    download_embedding_model(EMBEDDING_MODEL, CACHE_DIR)
    download_reranker_model(RERANKER_MODEL, CACHE_DIR)
    download_semantic_chunking_model(SEMANTIC_CHUNKING_MODEL, CACHE_DIR)
    download_conversation_model(CONVERSACTION_MODEL, CACHE_DIR)
