# Vessel-LLM

A powerful Chinese/English document processing and RAG (Retrieval-Augmented Generation) system with web interface. This system supports multiple document formats, intelligent document processing, and provides both traditional Q&A and RAG-enhanced responses.

## Features

- 🚀 **Multi-format Document Support**: Process DOCX, PDF, Excel, and TXT files
- 🤖 **Advanced RAG System**: Train custom knowledge bases and get contextual answers
- 🌐 **Web Interface**: User-friendly web interface for document upload and Q&A
- 💬 **Streaming Responses**: Real-time response streaming for better user experience
- 🧠 **Context Memory**: Maintains conversation context across sessions
- 📊 **Multiple RAG Models**: Support for multiple trained RAG models
- 🔄 **Real-time Training Progress**: Live progress updates during RAG training
- 🎯 **Chinese/English Optimized**: Specially optimized for Chinese and English documents

## System Requirements

- Python 3.8 or higher
- 16GB+ RAM (24GB+ recommended for large documents)
- GPU support recommended for faster model inference
- LM Studio running locally (for LLM inference)
- 12 GB vRAM (16+ recommended for large documents)
## Installation Guide

### Step 1: Create Virtual Environment

First, create and activate a Python virtual environment:

**Windows:**
```bash
# Create virtual environment
python -m venv vessel_llm_env

# Activate virtual environment
vessel_llm_env\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv vessel_llm_env

# Activate virtual environment
source vessel_llm_env/bin/activate
```

### Step 2: Install Dependencies

Install all required libraries from requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 3: Download Models

Download the required AI models to local cache:

```bash
python download_models_to_cache.py
```

This will download:
- Qwen3-Embedding-0.6B (embedding model)
- bge-reranker-v2-m3 (reranking model)
- gte-multilingual-reranker-base (semantic chunking model)

**Note**: This step may take 10-30 minutes depending on your internet speed.

### Step 4: Setup LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (recommended: Qwen3-14B or similar)
3. Start the local server in LM Studio on `http://localhost:1234`

### Step 5: Run the Server

Start the Vessel-LLM server:

```bash
python server.py
```

The server will start on `http://localhost:5000`

### Step 6: Open Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

You should see the Vessel-LLM web interface with options for:
- Document upload and Q&A
- RAG training and management
- Regular chat functionality

## Usage Guide

### Basic Q&A

1. Navigate to the main interface
2. Type your question in the text area
3. Click "Send" to get an AI response
4. Responses are streamed in real-time

### Document Processing

1. Click "Choose Files" to select documents
2. Supported formats: .docx, .pdf, .xlsx, .txt
3. Enter your question about the documents
4. Click "Upload and Ask" to process documents and get answers

### RAG Training

1. Click "Upload Folder for RAG Training"
2. Select multiple documents to create a knowledge base
3. Enter a name for your RAG model
4. Monitor training progress in real-time
5. Once trained, select your model from the dropdown to use it

### RAG Q&A

1. Select a trained RAG model from the dropdown
2. Ask questions related to your knowledge base
3. Get contextual answers based on your documents

## Project Structure

```
Vessel-LLM/
├── server.py                   # Main Flask server
├── rag_trainer.py             # RAG system implementation
├── rag_interface.py           # RAG interface module
├── document_extractor.py      # Document processing utilities
├── download_models_to_cache.py # Model download script
├── requirements.txt           # Python dependencies
├── index.html                # Web interface
├── styles.css               # Interface styling
├── script.js                # Frontend JavaScript
├── uploads/                 # Temporary file storage
├── rag_models/             # Trained RAG models
└── cache/                  # Model cache directory
```

## API Endpoints

- `POST /ask-stream` - Streaming Q&A
- `POST /upload_and_ask` - Document upload and Q&A
- `POST /rag_ask` - RAG-enhanced Q&A
- `POST /upload_folder_for_rag` - RAG training
- `GET /rag_models` - List available RAG models
- `POST /rename_rag_model` - Rename RAG model
- `POST /delete_rag_model` - Delete RAG model
- `GET /health` - Health check

## Configuration

### Model Configuration

Edit the model names in `download_models_to_cache.py` if you want to use different models:

```python
EMBEDDING_MODEL = "qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SEMANTIC_CHUNKING_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"
```

### LM Studio Configuration

Update the API endpoint in `server.py` if your LM Studio runs on a different port:

```python
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
```

## Troubleshooting

### Common Issues

1. **"Failed to connect to LLM API"**
   - Ensure LM Studio is running
   - Check that the model is loaded in LM Studio
   - Verify the API endpoint URL

2. **"RAG system is not available"**
   - Run `python download_models_to_cache.py` to download models
   - Check that models are downloaded to the cache directory

3. **"No documents found to train the system"**
   - Ensure uploaded documents are in supported formats
   - Check that documents contain readable text

4. **Memory Issues**
   - Use smaller batch sizes for large documents
   - Ensure sufficient RAM (8GB+ recommended)
   - Consider using CPU instead of GPU for large models

### Performance Tips

- Use GPU if available for faster inference
- Process documents in smaller batches
- Monitor memory usage during training
- Use SSD storage for better I/O performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace for the transformer models
- LlamaIndex for the RAG framework
- OpenAI for inspiration from ChatGPT
- The open-source AI community
