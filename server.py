from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
import html
from docx import Document
import PyPDF2
from openpyxl import load_workbook
import os
import logging
import shutil
from datetime import datetime
from collections import defaultdict, deque
import uuid
import sys
import json

# Import RAG system - adding the RAG directory to the path
rag_path = r'C:\Users\zeyua\Desktop\Coding\RAG'
sys.path.append(rag_path)
try:
    # Import directly from rag_trainer.py
    from rag_trainer import ChineseRAGSystem
    
    # RAG models directory
    BASE_MODEL_DIR = r"C:\Users\zeyua\Desktop\Coding\RAG\rag_models"
    
    # Create a dictionary to store multiple RAG systems
    rag_systems = {}
    
    # Check if default model exists and load it
    default_model_dir = os.path.join(BASE_MODEL_DIR, "default")
    if os.path.exists(default_model_dir) and os.path.exists(os.path.join(default_model_dir, "config.json")):
        rag_systems["default"] = ChineseRAGSystem(model_save_dir=default_model_dir)
        rag_systems["default"].load_system()
        print("Default RAG system loaded successfully!")
    
    # Load any other existing models
    for model_dir in os.listdir(BASE_MODEL_DIR):
        model_path = os.path.join(BASE_MODEL_DIR, model_dir)
        config_path = os.path.join(model_path, "config.json")
        if os.path.isdir(model_path) and os.path.exists(config_path) and model_dir != "default":
            try:
                rag_systems[model_dir] = ChineseRAGSystem(model_save_dir=model_path)
                rag_systems[model_dir].load_system()
                print(f"RAG system '{model_dir}' loaded successfully!")
            except Exception as e:
                print(f"Error loading RAG system '{model_dir}': {e}")
    
    RAG_AVAILABLE = True
    print(f"Loaded {len(rag_systems)} RAG systems")
except Exception as e:
    print(f"Error initializing RAG systems: {e}")
    rag_systems = {}
    RAG_AVAILABLE = False

app = Flask(__name__)
CORS(app) 

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# LM Studio API endpoint
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# Add a helper function to call the LLM API with proper error handling
def call_llm_api(messages, temperature=0.5, max_tokens=32000):
    """Helper function to call the LLM API with consistent error handling"""
    try:
        payload = {
            "model": "qwen3-14b", # This can be any name, LM Studio will use whatever model is loaded
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.info(f"Sending request to LLM API: {LM_STUDIO_API_URL}")
        logger.debug(f"Payload: {payload}")
        
        response = requests.post(
            LM_STUDIO_API_URL, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=120  # Increase timeout for longer responses
        )
        
        logger.info(f"LLM API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None, f"LLM API error: {response.text}"
            
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            logger.error(f"No choices in LLM response: {response_data}")
            return None, "No response from LLM"
            
        return response_data, None
        
    except requests.exceptions.ConnectionError:
        error_msg = "Failed to connect to LLM API. Is LM Studio running?"
        logger.error(error_msg)
        return None, error_msg
        
    except Exception as e:
        logger.error(f"Unexpected error calling LLM API: {str(e)}")
        return None, f"Error: {str(e)}"

def extract_docx_content(file_path):
    """Extract text from a .docx file."""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"Error extracting .docx: {str(e)}")
        return f"Error extracting .docx content: {str(e)}"

def extract_pdf_content(file_path):
    """Extract text from a .pdf file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join([reader.pages[i].extract_text() or "" for i in range(len(reader.pages))])
    except Exception as e:
        logger.error(f"Error extracting .pdf: {str(e)}")
        return f"Error extracting .pdf content: {str(e)}"

def extract_excel_content(file_path):
    """Extract text from a .xlsx file."""
    try:
        wb = load_workbook(file_path, read_only=True)
        content = []
        for sheet in wb:
            for row in sheet.rows:
                row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                content.append(" ".join(row_data))
        return "\n".join(content)
    except Exception as e:
        logger.error(f"Error extracting .xlsx: {str(e)}")
        return f"Error extracting .xlsx content: {str(e)}"

def extract_txt_content(file_path):
    """Extract text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting .txt: {str(e)}")
        return f"Error extracting .txt content: {str(e)}"

def extract_file_content(file_path, file_extension):
    """Extract content based on file extension."""
    if not os.path.exists(file_path):
        return "Error: File not found"
    
    try:
        if file_extension == '.docx':
            return extract_docx_content(file_path)
        elif file_extension == '.pdf':
            return extract_pdf_content(file_path)
        elif file_extension == '.xlsx':
            return extract_excel_content(file_path)
        elif file_extension == '.txt':
            return extract_txt_content(file_path)
        else:
            return "Error: Unsupported file format"
    except Exception as e:
        logger.error(f"Error in extract_file_content: {str(e)}")
        return f"Error extracting file content: {str(e)}"


@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle text-only questions (stateless, no chat memory)."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # System prompt for tool usage
        system_prompt = (
            "你是一个友善的助手"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        response_data, error = call_llm_api(messages)
        if error:
            return jsonify({'error': error}), 500

        full_response = response_data['choices'][0]['message']['content']

        # Try to find a JSON tool call anywhere in the output
        import json as _json
        tool_result = None
        tool_json = None
        tool_message = full_response
        tool_pattern = r'\{\s*"tool"\s*:\s*"[^"]+".*?\}'
        match = re.search(tool_pattern, full_response, re.DOTALL)
        if match:
            try:
                tool_json = _json.loads(match.group())
                if isinstance(tool_json, dict) and 'tool' in tool_json and 'args' in tool_json:
                    tool_name = tool_json['tool']
                    args = tool_json['args']
                    pass
            except Exception:
                pass  

        # If not a tool call, return normal LLM response
        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else f"Processing question: '{question}'. Analyzing intent..."
        response_text = full_response[think_match.end():].strip() if think_match else full_response

        result = {'thinking': thinking, 'response': response_text}
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_and_ask', methods=['POST'])
def upload_and_ask():
    """Handle questions with optional multiple file uploads (stateless, no chat memory)."""
    try:
        # Debug logging for file upload
        logger.debug(f"request.files keys: {list(request.files.keys())}")
        logger.debug(f"request.files.getlist('file') length: {len(request.files.getlist('file'))}")
        logger.debug(f"request.form: {request.form}")
        question = request.form.get('question', '')
        files = request.files.getlist('file')
        file_contents = []

        if files:
            os.makedirs('uploads', exist_ok=True)
            for file in files:
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in ['.docx', '.pdf', '.xlsx', '.txt']:
                    logger.error(f"Unsupported file format: {file_extension}")
                    return jsonify({'error': f"Unsupported file format: {file_extension}"}), 400

                # Save file temporarily
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)

                # Extract file content
                file_content = extract_file_content(file_path, file_extension)
                if file_content.startswith("Error"):
                    logger.error(f"File extraction failed for {file.filename}: {file_content}")
                    os.remove(file_path)
                    return jsonify({'error': file_content}), 400
                file_contents.append(f"File: {file.filename}\n{file_content}")

                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)

        # Combine question and file contents
        combined_file_content = "\n\n".join(file_contents) if file_contents else ""
        combined_input = f"Question: {question}\n\nFile Contents (if any):\n{combined_file_content}" if combined_file_content else question
        if not combined_input.strip():
            logger.error("No question or file content provided")
            return jsonify({'error': 'No question or file content provided'}), 400

        messages = [
            {"role": "system", "content": "你是一个友善的助手，你必须基于提供的信息来回答问题。请在 <think> 标签中提供你的推理。"},
            {"role": "user", "content": combined_input}
        ]

        response_data, error = call_llm_api(messages)
        if error:
            return jsonify({'error': error}), 500

        full_response = response_data['choices'][0]['message']['content']
        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else f"Processing input with question: '{question}'. Analyzing intent and file content..."
        response_text = full_response[think_match.end():].strip() if think_match else full_response

        result = {'thinking': thinking, 'response': response_text, 'format': 'text'}
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /upload_and_ask: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/log_user_message', methods=['POST'])
def log_user_message():
    """Log user input message with timestamp to a txt file."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"{timestamp} | {message}\n"
        
        # 使用相对路径，避免硬编码错误的用户路径
        log_path = os.path.join(os.path.dirname(__file__), 'user_message_log.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error in /log_user_message: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag_ask', methods=['POST'])
def rag_ask():
    """Handle questions using RAG system."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        model_id = data.get('model_id', 'default')  # Get model ID, default to "default"
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        if not RAG_AVAILABLE or not rag_systems:
            return jsonify({'error': 'RAG system is not available'}), 503
            
        # Check if the requested model exists
        if model_id not in rag_systems:
            return jsonify({'error': f"RAG model '{model_id}' not found"}), 404
            
        # Log that we're using RAG
        logger.info(f"Using RAG system '{model_id}' for question: {question}")
        
        # Process using RAG system
        try:
            # Generate answer using selected RAG model
            full_response = rag_systems[model_id].generate_answer(question, api_url=LM_STUDIO_API_URL)
            
            # Parse <think> tags
            think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            thinking = think_match.group(1).strip() if think_match else f"Processing question with RAG system '{model_id}': '{question}'. Searching knowledge base..."
            response_text = full_response[think_match.end():].strip() if think_match else full_response
            
            # Return the results
            result = {'thinking': thinking, 'response': response_text}
            return jsonify(result)
        except Exception as rag_error:
            logger.error(f"RAG processing error: {str(rag_error)}")
            return jsonify({'error': f"RAG processing error: {str(rag_error)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in /rag_ask: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a route to get available RAG models
@app.route('/rag_models', methods=['GET'])
def get_rag_models():
    """Get list of available RAG models."""
    try:
        if not RAG_AVAILABLE:
            return jsonify({'error': 'RAG system is not available'}), 503
        
        models = list(rag_systems.keys())
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Error getting RAG models: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Process uploaded folder for RAG training
def process_folder_for_rag(upload_dir):
    """Process files in a directory for RAG training."""
    processed_dir = os.path.join(upload_dir, "processed_texts")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process all supported files in the directory
    file_count = 0
    for root, _, files in os.walk(upload_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            # Skip already processed files
            if "processed_texts" in file_path:
                continue
                
            if file_extension in ['.txt', '.docx', '.pdf', '.xlsx']:
                try:
                    # Extract content
                    content = extract_file_content(file_path, file_extension)
                    if not content.startswith("Error"):
                        # Save as processed text file
                        processed_file = os.path.join(processed_dir, f"{os.path.splitext(file)[0]}_processed.txt")
                        with open(processed_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        file_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
    
    return processed_dir, file_count

# Add a route to handle folder upload and RAG training
@app.route('/upload_folder_for_rag', methods=['POST'])
def upload_folder_for_rag():
    """Handle folder upload and train RAG model."""
    try:
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
            
        # Get model name from form data
        model_name = request.form.get('model_name', '')
        if not model_name:
            model_name = f"rag_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize model name (only allow alphanumeric, underscore and hyphen)
        model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        
        # Create directory for this upload
        upload_dir = os.path.join('uploads', f"rag_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save all uploaded files
        files = request.files.getlist('files[]')
        saved_paths = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Preserve directory structure by splitting path
            filepath_parts = file.filename.split('/')
            if len(filepath_parts) > 1:
                # Create subdirectories if needed
                subdir = os.path.join(upload_dir, *filepath_parts[:-1])
                os.makedirs(subdir, exist_ok=True)
                save_path = os.path.join(upload_dir, *filepath_parts)
            else:
                save_path = os.path.join(upload_dir, file.filename)
                
            file.save(save_path)
            saved_paths.append(save_path)
            
        logger.info(f"Saved {len(saved_paths)} files for RAG training")
        
        # Process files
        processed_dir, file_count = process_folder_for_rag(upload_dir)
        
        if file_count == 0:
            return jsonify({'error': 'No valid files for processing'}), 400
            
        # Create model directory
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Train RAG model
        try:
            # Create and train a new RAG system
            new_rag = ChineseRAGSystem(processed_texts_dir=processed_dir, model_save_dir=model_dir)
            new_rag.train_system(processed_dir)
            
            # Store the new model
            rag_systems[model_name] = new_rag
            
            # Clean up upload directory but keep processed texts
            processed_texts_backup = os.path.join(model_dir, "processed_texts")
            if os.path.exists(processed_texts_backup):
                shutil.rmtree(processed_texts_backup)
            shutil.copytree(processed_dir, processed_texts_backup)
            
            # Don't delete upload_dir for safety - can be cleaned up later or manually
            
            return jsonify({
                'success': True,
                'model_name': model_name,
                'processed_files': file_count
            })
        except Exception as e:
            logger.error(f"RAG training error: {str(e)}")
            return jsonify({'error': f"RAG training failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_folder_for_rag: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a simple health check endpoint to test if the server is running
@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to check if server is running"""
    return jsonify({"status": "ok", "llm_url": LM_STUDIO_API_URL, "rag_available": RAG_AVAILABLE})

# Add an endpoint to test LLM connection
@app.route('/test_llm', methods=['GET'])
def test_llm():
    """Test connection to the LLM API"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
        
        response_data, error = call_llm_api(messages, max_tokens=50)
        if error:
            return jsonify({"status": "error", "message": error}), 500
            
        return jsonify({
            "status": "ok", 
            "message": "LLM connection successful", 
            "response": response_data['choices'][0]['message']['content']
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # 确保uploads目录存在
    os.makedirs('uploads', exist_ok=True)
    # 检查LLM是否可用
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test"}
        ]
        test_response, test_error = call_llm_api(test_messages, max_tokens=10)
        if test_error:
            logger.warning(f"LLM API not available at startup: {test_error}")
            print(f"Warning: LLM API not available. Please make sure LM Studio is running at {LM_STUDIO_API_URL}")
        else:
            logger.info("LLM API available and responding")
            print(f"LLM API available at {LM_STUDIO_API_URL}")
    except Exception as e:
        logger.error(f"Error testing LLM connection at startup: {str(e)}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)