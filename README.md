# 📌 Note: Summer Intern Version

> If you're looking for the **Summer Intern version** (lightweight, experimental, or early prototype features), please switch to the `summer_intern` branch.  
> Example:  
> ```bash
> git clone -b summer_intern https://github.com/Anderson-ops-oss/Vessel-LLM.git
> ```

---

# Update Log - Qwen3-VL Integration Branch (2026-01-21)

## 🚨 Critical Hardware Warning 

**Current Status: EXTREME RESOURCE USAGE **

This branch contains experimental integrations of the Qwen3-VL family models. The current implementation imposes massive requirements on system resources.

**Tested Environment:**
- **VRAM**: 22 GB (RTX 2080 Ti Modified / Quadro equivalent) -> **FAILED (OOM / Full)**
- **System RAM**: 48 GB -> **FAILED (Full Saturation)**

**Conclusion:**
The current setup (running Qwen3-VL-8B-Thinking + Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B simultaneously) is **NOT executable** on consumer hardware with < 24GB VRAM, and likely struggles even on an RTX 4090 (24GB).
---

## 🛠️ Technical Changes 
### 1. Backend Architecture
- **Removed LM Studio Dependency**: The system no longer relies on an external LM Studio API server.
- **Local Inference Engine**: Integrated `transformers` and `accelerate` directly into `server.py` for native Python inference.
- **Streaming Refactor**: Re-implemented streaming using `TextIteratorStreamer` with multi-threading to support real-time token generation from local models.

### 2. Model Upgrades
- **Main LLM**: Updated to `Qwen/Qwen3-VL-8B-Thinking` (configured for 4-bit quantization via `bitsandbytes`).
- **Multimodal Support**: Backend now supports processing Images and PDF-to-Image conversion directly for visual question answering.

### 3. RAG System Overhaul
- **Custom Embedding Adapter**:
  - Implemented `Qwen3VLEmbedding` class to bypass LlamaIndex's incompatibility with `Qwen3VLConfig`.
  - Manually implements Mean Pooling to extract dense vectors from the generative model.
- **Custom Reranker Adapter**:
  - Implemented `Qwen3VLReranker` class.
  - Due to the generative nature of Qwen3-VL (lack of classification head), the Reranker now operates in a **Bi-Encoder mode** (Cosine Similarity of embeddings) instead of a traditional Cross-Encoder mode.

### 4. Optimization Attempts
- **Singleton Model Loading**: Refactored `server.py` to load Embedding and Reranker models only **once** globally, sharing the instance across all RAG knowledge bases to prevent exponential VRAM usage.
- **Mixed Precision**:
  - Main LLM: 4-bit Quantization (NF4).
  - RAG Models: FP16 (Standard Precision) - *Note: This is the current bottleneck causing high VRAM usage.*

## 📝 Known Issues 
1. **OOM / High VRAM**: The combined footprint of one 8B LLM and two 2B FP16 models exceeds 22GB VRAM.
2. **System RAM Spike**: Loading `bitsandbytes` models and `AutoModel` simultaneously spikes system RAM usage, potentially causing crashes on systems with < 64GB RAM.
3. **Quantization Incompatibility**: 4-bit quantization for the RAG models (Embedding/Reranker) caused `weight is not an nn.Module` errors due to conflicts with Qwen3-VL's custom remote code, forcing a rollback to FP16.

## 🔜 Next Steps 
- Investigate lower precision (4B model) or alternative embedding models to fit consumer hardware.
- Explore 8-bit loading for RAG models or fix the 4-bit compatibility issue.