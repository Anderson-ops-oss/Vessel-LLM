import os
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "qwen/Qwen3-Embedding-0.6B" # 更新为8B版本以提高准确率
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" # 更新为BGE-Reranker-v2-M3
SEMANTIC_CHUNKING_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base" # GTE Base模型用于语义分块


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

# 自动查找 snapshots 绝对路径
import glob
def find_snapshot(model_name):
    # model_name 形如 "qwen/Qwen3-Embedding-0.6B"，转换为 models--qwen--Qwen3-Embedding-0.6B
    repo = model_name.replace('/', '--')
    pattern = os.path.join(CACHE_DIR, f"models--{repo}", "snapshots", "*")
    paths = glob.glob(pattern)
    if paths:
        return os.path.abspath(paths[0])
    else:
        return None

embedding_path = find_snapshot(EMBEDDING_MODEL)
reranker_path = find_snapshot(RERANKER_MODEL)
semantic_chunking_path = find_snapshot(SEMANTIC_CHUNKING_MODEL)

print("\n本地模型路径：")
print(f"Embedding model local path: {embedding_path if embedding_path else '未找到'}")
print(f"Reranker model local path: {reranker_path if reranker_path else '未找到'}")
print(f"Semantic chunking model local path: {semantic_chunking_path if semantic_chunking_path else '未找到'}")
print("\n请将这些路径填入 rag_trainer.py 的 embedding_model/semantic_chunking_model/reranker_model 配置，断网即可离线加载！")

print("All models downloaded!")

