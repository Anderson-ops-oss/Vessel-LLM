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

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB+ recommended for large documents)
- **Storage**: 10GB free space (for models and cache)
- **Internet**: Required for initial model download
- **LM Studio**: Latest version running locally

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 24GB+ (32GB+ for optimal performance)
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **Storage**: 20GB+ free space (SSD recommended)
- **CPU**: Multi-core processor (Intel i7/AMD Ryzen 7 or better)

### Platform Support
- ✅ **Windows 10/11** (Primary support)

### External Dependencies
- **LM Studio**: For LLM inference ([Download here](https://lmstudio.ai/))
- **Git**: For cloning the repository (optional)
## Installation Guide

### Step 1: Clone the Repository

First, clone the Vessel-LLM repository from GitHub:

```bash
git clone https://github.com/Anderson-ops-oss/Vessel-LLM.git
cd Vessel-LLM
```

### Step 2: Create Virtual Environment

Create and activate a Python virtual environment using your preferred method:

#### Option A: Using Python venv

**Windows:**
```bash
# Create virtual environment
python -m venv vessel_llm_env

# Activate virtual environment
vessel_llm_env\Scripts\activate
```

#### Option B: Using Anaconda/Miniconda

**All Platforms:**
```bash
# Create conda environment with Python 3.9
conda create -n vessel_llm python=3.9

# Activate conda environment
conda activate vessel_llm
```

#### Option C: Using pipenv

```bash
# Install pipenv if not already installed
pip install pipenv

# Create and activate environment
pipenv install
pipenv shell
```

### Step 3: Install Dependencies

Install all required libraries from requirements.txt:

```bash
pip install -r requirements.txt
```

**Note**: This may take 1-5 minutes depending on your internet speed and system.

### Step 4: Download AI Models

Download the required AI models to local cache:

```bash
python scripts/download_models_to_cache.py
```

This will download:
- **Qwen3-Embedding-0.6B** (embedding model)
- **bge-reranker-v2-m3** (reranking model)  
- **bge-m3** (semantic chunking model)

**Note**: This step may take 10-30 minutes depending on your internet speed. The models will be cached locally for future use.

### Step 5: Setup LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (recommended: qwen/qwen3-14b or similar)
3. Start the local server in LM Studio on `http://localhost:1234`
4. Ensure the model is loaded and the server is running

### Step 6: Run the Server

Start the Vessel-LLM server:

```bash
python server.py
```

The server will start on `http://localhost:5000`

### Step 7: Open Web Interface

Open your index.html

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
2. Supported formats: .docx, .pdf, .xlsx, .xls, .txt, .md
3. Enter your question about the documents
4. Click "Upload and Ask" to process documents and get answers

### RAG Training

1. Click "Upload Folder for RAG Training"
2. Select folder to create a knowledge base
3. Monitor training progress in real-time
4. Once trained, select your model from the dropdown to use it

### RAG Q&A

1. Select a trained RAG model from the dropdown
2. Ask questions related to your knowledge base
3. Get contextual answers based on your documents

## Deployment Guide

### Creating Standalone Executable

For creating a standalone executable that can be distributed without requiring Python installation:

#### Prerequisites

1. **Install PyInstaller**:
   ```bash
   pip install pyinstaller
   ```

2. **Install Inno Setup** (Windows only):
   - Download from [Inno Setup official website](https://jrsoftware.org/isinfo.php)
   - Install the software on your Windows machine

#### Step 1: Build Executable Files

Build the main server executable:

```bash
python -m PyInstaller server.spec
```

Build the model downloader executable:

```bash
python -m PyInstaller download_models_to_cache.spec
```

These commands will create:
- `dist/server/` - Contains the main application executable
- `dist/download_models_to_cache/` - Contains the model downloader executable

#### Step 2: Create Windows Installer

Use Inno Setup to create a professional Windows installer:

1. **Open Inno Setup Compiler**
2. **Open the installer script**:
   ```
   File → Open → installer.iss
   ```

3. **Build the installer**:
   ```
   Build → Compile
   ```

4. **The installer will be created** in the `Output/` directory

#### Step 3: Distribution

The resulting installer (`Vessel-LLM-Setup.exe`) will:
- Install the application
- Create desktop shortcuts
- Handle all dependencies automatically
- Include the model downloader utility

### Manual Deployment Options

#### Option A: Portable Distribution

Create a portable version by copying the `dist/` folder contents:

1. Copy `dist/server/` folder to target machine
2. Copy `dist/download_models_to_cache/` folder 
3. Run `download_models_to_cache.exe` first to download models
4. Run `server.exe` to start the application


### Deployment Notes

- **First Run**: Always run the model downloader first on the target machine
- **Internet Required**: Initial setup requires internet connection for model downloads
- **System Requirements**: Ensure target machines meet minimum system requirements
- **LM Studio**: Users still need to install and configure LM Studio separately
- **Antivirus**: Some antivirus software may flag the executable - add exceptions if needed

## Project Structure

```
Vessel-LLM/
├── server.py                      # Main Flask server
├── server.spec                    # PyInstaller spec for server
├── download_models_to_cache.spec   # PyInstaller spec for model downloader
├── installer.iss                  # Inno Setup installer script
├── requirements.txt               # Python dependencies
├── start.bat                      # Windows startup script
├── README.md                      # Project documentation
├── core/                          # Core application modules
│   ├── document_extractor.py      # Document processing utilities
│   ├── rag_trainer.py             # RAG system implementation
│   └── rag_interface.py           # RAG interface module (unused)
├── scripts/                       # Utility scripts
│   └── download_models_to_cache.py # Model download script
├── web_folder/                    # Web interface files
│   ├── index.html                 # Main web interface
│   ├── css/
│   │   └── styles.css             # Interface styling
│   ├── img/                       # Images and icons
│   │   ├── logo.ico
│   │   └── OOCL_logo_slogan.png
│   └── js/
│       └── script.js              # Frontend JavaScript
├── uploads/                       # Temporary file storage (created at runtime)
├── rag_models/                    # Trained RAG models (created at runtime)
├── cache/                         # Model cache directory (created at runtime)
└── dist/                          # Built executables (created by PyInstaller)
    ├── server/                    # Server executable distribution
    └── download_models_to_cache/  # Model downloader distribution
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

Edit the model names in `scripts/download_models_to_cache.py` if you want to use different models:

```python
EMBEDDING_MODEL = "qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SEMANTIC_CHUNKING_MODEL = "BAAI/bge-m3"
```

### LM Studio Configuration

Update the API endpoint in `server.py` if your LM Studio runs on a different port:

```python
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
```

### Environment Variables

The application automatically sets these environment variables:

```python
HF_HOME = "./cache"                    # HuggingFace cache directory
TRANSFORMERS_CACHE = "./cache"         # Transformers cache
HF_DATASETS_CACHE = "./cache"         # Datasets cache
SENTENCE_TRANSFORMERS_HOME = "./cache" # Sentence transformers cache
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
   - Ensure sufficient RAM (24GB+ recommended)
   - Consider using CPU instead of GPU for large models

### Performance Tips

- Use GPU if available for faster inference
- Process documents in smaller batches
- Monitor memory usage during training
- Use SSD storage for better I/O performance


## Acknowledgments

- HuggingFace for the transformer models
- LlamaIndex for the RAG framework
- The open-source AI community
