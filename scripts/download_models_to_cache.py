import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

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
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")
    SEMANTIC_CHUNKING_MODEL = os.getenv("SEMANTIC_CHUNKING_MODEL", "")
    CONVERSACTION_MODEL = os.getenv("LLM_MODEL", "")

    if not all([EMBEDDING_MODEL, RERANKER_MODEL, SEMANTIC_CHUNKING_MODEL, CONVERSACTION_MODEL]):
        raise ValueError("One or more model names are not set in environment variables. Make sure you have set EMBEDDING_MODEL, RERANKER_MODEL, SEMANTIC_CHUNKING_MODEL, and LLM_MODEL in .env file.")
    
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
