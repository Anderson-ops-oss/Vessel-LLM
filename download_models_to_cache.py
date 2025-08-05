import os
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "qwen/Qwen3-Embedding-0.6B" 
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" 
SEMANTIC_CHUNKING_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base" 


# 指定 cache 路径（自动获取绝对路径，适配不同环境）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'cache'))
os.makedirs(CACHE_DIR, exist_ok=True)

# 设置环境变量，确保 transformers 下载到指定 cache
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR

# 下载 embedding 模型
print(f"Downloading embedding model: {EMBEDDING_MODEL} ...")
AutoModel.from_pretrained(EMBEDDING_MODEL, cache_dir=CACHE_DIR)
AutoTokenizer.from_pretrained(EMBEDDING_MODEL, cache_dir=CACHE_DIR)
print(f"Embedding model downloaded to {CACHE_DIR}")

print(f"Downloading reranker model: {RERANKER_MODEL} ...")
AutoModel.from_pretrained(RERANKER_MODEL, cache_dir=CACHE_DIR)
AutoTokenizer.from_pretrained(RERANKER_MODEL, cache_dir=CACHE_DIR)
print(f"Reranker model downloaded to {CACHE_DIR}")

print(f"Downloading semantic chunking model: {SEMANTIC_CHUNKING_MODEL} ...")
AutoModel.from_pretrained(SEMANTIC_CHUNKING_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True)
AutoTokenizer.from_pretrained(SEMANTIC_CHUNKING_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True)
print(f"Semantic chunking model downloaded to {CACHE_DIR}")
