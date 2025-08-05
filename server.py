
import os
# 设置 HuggingFace 及相关模型缓存目录到 main_function 目录下的 cache 子目录（自动绝对路径，跨环境可用）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = os.path.abspath(CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = os.path.abspath(CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.abspath(CACHE_DIR)

from flask import Flask, request, jsonify, Response, stream_with_context, session
from flask_cors import CORS
import os
# 设置 HuggingFace 及相关模型缓存目录到 main_function 目录下的 cache 子目录（自动绝对路径，跨环境可用）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = os.path.abspath(CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = os.path.abspath(CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.abspath(CACHE_DIR)

from flask import Flask, request, jsonify, Response, stream_with_context, session
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
from flask import Response
import time
from pathlib import Path
import uuid

# Import document processor
from document_extractor import ChineseDocumentProcessor

# Import RAG system - adding the RAG directory to the path
rag_path = r'main_function'
sys.path.append(rag_path)
try:
    # Import directly from rag_trainer.py
    from rag_trainer import ChineseRAGSystem
    
    # RAG models directory
    BASE_MODEL_DIR = r"rag_models"

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
app.secret_key = 'your-secret-key-change-this-in-production'

# Store conversation contexts for each session
conversation_contexts = defaultdict(lambda: deque(maxlen=20))  # Keep last 20 messages 

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def format_markdown_to_user_friendly(text):
    """Convert markdown text to user-friendly plain text format."""
    if not text:
        return text
    
    # Store original text for comparison
    original_text = text
    
    # Remove markdown headers and replace with clean formatting
    text = re.sub(r'^###\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*40, text, flags=re.MULTILINE)
    text = re.sub(r'^##\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*50, text, flags=re.MULTILINE)
    text = re.sub(r'^#\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*60, text, flags=re.MULTILINE)
    
    # Handle regular headers without bold
    text = re.sub(r'^###\s+(.*?)$', r'\n\1\n' + '-'*30, text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'\n\1\n' + '-'*40, text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'\n\1\n' + '-'*50, text, flags=re.MULTILINE)
    
    # Remove bold formatting but keep the content
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove italic formatting but keep the content
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Format numbered lists more clearly with proper spacing
    text = re.sub(r'^(\d+)\.\s+\*\*(.*?)\*\*\s*$', r'\n\1. \2', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\.\s+(.*?)$', r'\n\1. \2', text, flags=re.MULTILINE)
    
    # Format bullet points with better spacing
    text = re.sub(r'^[\*\-\+]\s+\*\*(.*?)\*\*\s*$', r'\n• \1', text, flags=re.MULTILINE)
    text = re.sub(r'^[\*\-\+]\s+(.*?)$', r'\n• \1', text, flags=re.MULTILINE)
    
    # Remove code blocks backticks but keep the content with better formatting
    text = re.sub(r'```.*?\n(.*?)```', r'\n[\1]\n', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'[\1]', text)
    
    # Handle special formatting patterns
    text = re.sub(r'\*\*Steps:\*\*', 'Steps:', text)
    text = re.sub(r'\*\*Note:\*\*', 'Note:', text)
    text = re.sub(r'\*\*Important:\*\*', 'Important:', text)
    
    # Clean up multiple newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove extra spaces while preserving single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Clean up lines and maintain readable structure
    lines = text.split('\n')
    formatted_lines = []
    prev_was_empty = False
    
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(line)
            prev_was_empty = False
        elif not prev_was_empty and formatted_lines:  # Keep single empty lines between content
            formatted_lines.append('')
            prev_was_empty = True
    
    # Join lines and clean up
    formatted_text = '\n'.join(formatted_lines).strip()
    
    # If formatting didn't improve readability much, return original with minimal changes
    if len(formatted_text) < len(original_text) * 0.5:
        # Just remove the most aggressive markdown and return
        simple_format = re.sub(r'\*\*(.*?)\*\*', r'\1', original_text)
        simple_format = re.sub(r'`([^`]+)`', r'\1', simple_format)
        return simple_format
    
    return formatted_text


# LM Studio API endpoint
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 日志用户问题
def log_user_question(question, session_id=None):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"{timestamp} | session: {session_id or '-'} | {question}\n"
        log_path = os.path.join(os.path.dirname(__file__), 'user_question_log.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
    except Exception as e:
        logger.error(f"Failed to log user question: {e}")

# Add a helper function to call the LLM API with proper error handling
def call_llm_api(messages, temperature=0.5, max_tokens=32000, stream=False):
    """Helper function to call the LLM API with consistent error handling"""
    try:
        payload = {
            "model": "qwen/qwen3-14b", 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        logger.info(f"Sending request to LLM API: {LM_STUDIO_API_URL}")
        logger.debug(f"Payload: {payload}")
        
        response = requests.post(
            LM_STUDIO_API_URL, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=300,
            stream=stream
        )
        
        logger.info(f"LLM API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None, f"LLM API error: {response.text}"
        
        if stream:
            return response, None
        else:
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

def build_conversation_context(session_id, new_message, system_prompt=None):
    """Build conversation context with history"""
    context = conversation_contexts[session_id]
    
    # Add system message if provided and context is empty
    messages = []
    if system_prompt and (not context or context[0].get('role') != 'system'):
        messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    messages.extend(list(context))
    
    # Add new user message
    messages.append({"role": "user", "content": new_message})
    
    return messages

def update_conversation_context(session_id, user_message, ai_response):
    """Update conversation context with new messages"""
    context = conversation_contexts[session_id]
    context.append({"role": "user", "content": user_message})
    context.append({"role": "assistant", "content": ai_response})
    logger.info(f"Updated context for session_id: {session_id}, now has {len(context)} messages")

def get_or_create_session_id():
    """Get session ID from request or create new one"""
    # First try to get from request JSON data (only for POST/PUT requests with JSON content)
    if request.method in ['POST', 'PUT'] and request.is_json and request.json and 'session_id' in request.json:
        return request.json['session_id']
    
    # Then try to get from URL parameters (for GET requests)  
    if 'session_id' in request.args:
        return request.args['session_id']
    
    # Finally try to get from Flask session or create new one
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

# Remove old individual extraction functions and replace with unified processor
def extract_file_content(file_path, file_extension):
    """Extract content using the unified document processor."""
    logger.info(f"Extracting content from: {file_path} with extension: {file_extension}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return "Error: File not found"
    
    try:
        # Create a temporary directory for processing
        temp_dir = os.path.dirname(file_path)
        processor = ChineseDocumentProcessor(temp_dir)
        
        # Process the specific file based on extension
        if file_extension == '.docx':
            logger.info(f"Processing DOCX file: {file_path}")
            content = processor.extract_text_from_docx(file_path)
        elif file_extension == '.pdf':
            logger.info(f"Processing PDF file: {file_path}")
            content = processor.extract_text_from_pdf(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            logger.info(f"Processing Excel file: {file_path}")
            content = processor.extract_text_from_excel(file_path)
        elif file_extension == '.txt':
            logger.info(f"Processing TXT file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        else:
            logger.error(f"Unsupported file extension: {file_extension}")
            return "Error: Unsupported file format"
        
        if content:
            logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
            return content
        else:
            logger.warning(f"No content extracted from {file_path}")
            return "Error: No content extracted"
        
    except Exception as e:
        logger.error(f"Error in extract_file_content for {file_path}: {str(e)}", exc_info=True)
        return f"Error extracting file content: {str(e)}"

# 存储当前训练进度的全局变量
training_progress = {
    'percentage': 0,
    'status': '',
    'is_training': False,
    'error': None
}

# SSE 路由，用于实时推送训练进度
@app.route('/training_progress', methods=['GET'])
def training_progress_stream():
    def generate():
        while True:
            yield f"data: {json.dumps(training_progress)}\n\n"
            time.sleep(1)  # 每秒推送一次
    return Response(generate(), mimetype='text/event-stream')



@app.route('/ask-stream', methods=['POST'])
def ask_question_stream():
    """Handle streaming questions with context"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400


        # Get or create session ID
        session_id = get_or_create_session_id()
        logger.info(f"Using session_id: {session_id} for ask-stream")

        # 日志用户问题
        log_user_question(question, session_id)

        # System prompt for tool usage
        system_prompt = (
            "你是一个友善的助手。你可以参考之前的对话内容来提供更好的回答。"
            "请用清晰、简洁的语言回答，避免过多的标记符号。"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        # Build conversation context with history
        messages = build_conversation_context(session_id, question, system_prompt)

        def generate():
            ai_response_content = ""
            try:
                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        # Send raw content for streaming (format at end)
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the complete response before saving
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

    except Exception as e:
        logger.error(f"Error in /ask-stream: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload_and_ask', methods=['POST'])
def upload_and_ask():
    """Handle questions with optional multiple file uploads using streaming response and context memory."""
    try:
        # Debug logging for file upload
        logger.debug(f"request.files keys: {list(request.files.keys())}")
        logger.debug(f"request.files.getlist('file') length: {len(request.files.getlist('file'))}")
        logger.debug(f"request.form: {request.form}")
        question = request.form.get('question', '')
        files = request.files.getlist('file')
        file_contents = []


        # Get or create session ID
        session_id = request.form.get('session_id') or str(uuid.uuid4())

        # 日志用户问题
        log_user_question(question, session_id)
        
        logger.info(f"Upload and ask request - Question: {'YES' if question else 'NO'}, Files count: {len(files)}")

        if files:
            os.makedirs('uploads', exist_ok=True)
            for i, file in enumerate(files):
                # 检查文件是否有效
                if not file or not file.filename:
                    logger.warning(f"Skipping empty or invalid file at index {i}")
                    continue
                    
                logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension not in ['.docx', '.pdf', '.xlsx', '.txt']:
                    logger.error(f"Unsupported file format: {file_extension} for file: {file.filename}")
                    return jsonify({'error': f"Unsupported file format: {file_extension}"}), 400

                # Save file temporarily
                file_path = os.path.join('uploads', file.filename)
                logger.info(f"Saving file to: {file_path}")
                file.save(file_path)

                # Extract file content
                file_content = extract_file_content(file_path, file_extension)
                if file_content.startswith("Error"):
                    logger.error(f"File extraction failed for {file.filename}: {file_content}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({'error': file_content}), 400
                    
                logger.info(f"Successfully extracted content from {file.filename}: {len(file_content)} characters")
                file_contents.append(f"File: {file.filename}\n{file_content}")

                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            
            logger.info(f"Total files processed successfully: {len(file_contents)}")
        else:
            logger.info("No files provided in request")

        # Combine question and file contents
        combined_file_content = "\n\n".join(file_contents) if file_contents else ""
        combined_input = f"Question: {question}\n\nFile Contents (if any):\n{combined_file_content}" if combined_file_content else question
        if not combined_input.strip():
            logger.error("No question or file content provided")
            return jsonify({'error': 'No question or file content provided'}), 400

        # System prompt for file processing
        system_prompt = (
            "你是一个友善的助手，你必须基于提供的信息来回答问题。你可以参考之前的对话内容来提供更好的回答。"
            "请用清晰、简洁的语言回答，避免过多的标记符号。"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        # Build conversation context with history
        messages = build_conversation_context(session_id, combined_input, system_prompt)

        def generate():
            ai_response_content = ""
            try:
                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the response for better user experience
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)  # 只保存原始问题，不包含文件内容
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

    except Exception as e:
        logger.error(f"Error in /upload_and_ask: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/rag_ask', methods=['POST'])
def rag_ask():
    """Handle questions using RAG system with streaming response and context memory."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        model_id = data.get('model_id', 'default') 
        session_id = data.get('session_id') or str(uuid.uuid4())
        

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # 日志用户问题
        log_user_question(question, session_id)

        if not RAG_AVAILABLE or not rag_systems:
            return jsonify({'error': 'RAG system is not available'}), 503

        # Check if the requested model exists
        if model_id not in rag_systems:
            return jsonify({'error': f"RAG model '{model_id}' not found"}), 404

        # Log that we're using RAG
        logger.info(f"Using RAG system '{model_id}' for question: {question}")
        
        def generate():
            ai_response_content = ""
            try:
                # Get relevant context from RAG system
                try:
                    # Use RAG system to get context and generate response
                    rag_context = rag_systems[model_id].retrieve_relevant_docs(question)
                    
                    # Build enhanced prompt with RAG context and conversation history
                    system_prompt = (
                        f"你是一个友善的助手，基于提供的知识库内容和之前的对话历史来回答问题。"
                        f"请用清晰、简洁的语言回答，避免过多的标记符号。"
                        f"在回答之前，请在 <think> 标签中提供你的推理。\n\n"
                        f"相关知识库内容：\n{rag_context}"
                    )
                    
                    # Build conversation context with history
                    messages = build_conversation_context(session_id, question, system_prompt)
                    
                except Exception as rag_error:
                    logger.error(f"RAG retrieval error: {str(rag_error)}")
                    yield f"data: {json.dumps({'error': f'RAG retrieval error: {str(rag_error)}'})}\n\n"
                    return

                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the response for better user experience
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in RAG streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'RAG streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
            
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
    """Process files in a directory for RAG training using unified document processor."""
    processed_dir = os.path.join(upload_dir, "processed_texts")
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # 尝试检测和修复PDF文件
        for root, dirs, files in os.walk(upload_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    try:
                        # 尝试打开PDF文件验证其完整性
                        try:
                            with open(file_path, 'rb') as f:
                                # 尝试读取前1024字节检查PDF头
                                header = f.read(1024)
                                if not header.startswith(b'%PDF-'):
                                    logger.warning(f"File doesn't have PDF header: {file_path}")
                                    # 如果不是PDF格式，尝试重命名为txt文件
                                    txt_path = file_path.replace('.pdf', '.txt')
                                    os.rename(file_path, txt_path)
                                    logger.info(f"Renamed problematic PDF to text file: {txt_path}")
                        except Exception as e:
                            logger.error(f"Error checking PDF file {file_path}: {e}")
                    except Exception as check_error:
                        logger.error(f"Error during PDF check: {check_error}")

        # 使用统一文档处理器
        processor = ChineseDocumentProcessor(upload_dir)
        
        processed_texts = processor.process_documents()
        
        # 保存处理后的文本
        file_count = 0
        for file_path, text in processed_texts.items():
            if text.strip():  # 只保存非空文本
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                processed_file = os.path.join(processed_dir, f"{base_name}_processed.txt")
                
                with open(processed_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                file_count += 1
                logger.info(f"Processed and saved: {processed_file}")
        
        return processed_dir, file_count
        
    except Exception as e:
        logger.error(f"Error in process_folder_for_rag: {e}")
        return processed_dir, 0

# Add a route to handle folder upload and RAG training
# 修改 upload_folder_for_rag 路由以更新训练进度
@app.route('/upload_folder_for_rag', methods=['POST'])
def upload_folder_for_rag():
    """Handle folder upload and train RAG model."""
    global training_progress
    try:
        # 初始化训练进度
        training_progress = {
            'percentage': 10,
            'status': 'Uploading files...',
            'is_training': True,
            'error': None
        }

        # Check if files were uploaded
        if 'files[]' not in request.files:
            training_progress['error'] = 'No files provided'
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': 'No files provided'}), 400
            
        # Get folder name from form data (instead of user input)
        folder_name = request.form.get('folder_name', '')
        if not folder_name:
            # 如果没有提供文件夹名称，生成一个默认名称
            folder_name = f"rag_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize folder name - 允许中文字符、字母、数字和常用符号
        # 只过滤掉文件系统不允许的字符
        import string
        forbidden_chars = '<>:"/\\|?*'  # Windows文件系统禁止的字符
        model_name = folder_name
        for char in forbidden_chars:
            model_name = model_name.replace(char, '_')
        
        # 移除首尾空格并确保不为空
        model_name = model_name.strip()
        if not model_name:
            model_name = f"rag_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if model name already exists, add suffix if needed
        original_model_name = model_name
        counter = 1
        while model_name in rag_systems:
            model_name = f"{original_model_name}_{counter}"
            counter += 1
        
        # Create directory for this upload
        upload_dir = os.path.join('uploads', f"rag_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save all uploaded files
        files = request.files.getlist('files[]')
        saved_paths = []
        
        for file in files:
            if not file or not file.filename or file.filename == '':
                continue
                
            filepath_parts = file.filename.split('/')
            if len(filepath_parts) > 1:
                subdir = os.path.join(upload_dir, *filepath_parts[:-1])
                os.makedirs(subdir, exist_ok=True)
                save_path = os.path.join(upload_dir, *filepath_parts)
            else:
                save_path = os.path.join(upload_dir, file.filename)
                
            file.save(save_path)
            saved_paths.append(save_path)
            
        logger.info(f"Saved {len(saved_paths)} files for RAG training")
        
        # 检查文件类型并记录PDF文件
        pdf_files = []
        for path in saved_paths:
            if path.lower().endswith('.pdf'):
                pdf_files.append(path)
        
        # 更新进度
        training_progress['percentage'] = 10
        training_progress['status'] = f'Processing {len(saved_paths)} files (including {len(pdf_files)} PDFs)...'

        # 处理文件
        processed_dir, file_count = process_folder_for_rag(upload_dir)
        
        if file_count == 0:
            error_msg = 'No valid files for processing. PDF files might be corrupted or in an unsupported format.'
            training_progress['error'] = error_msg
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': error_msg}), 400
            
        # 更新进度
        training_progress['percentage'] = 30
        training_progress['status'] = f'Successfully processed {file_count} files. Training RAG model...'

        # Create model directory
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Train RAG model
        try:
            new_rag = ChineseRAGSystem(processed_texts_dir=processed_dir, model_save_dir=model_dir)
            new_rag.train_system(processed_dir)
            training_progress['percentage'] = 50
            training_progress['status'] = 'Training RAG model...'
            
            # Store the new model
            rag_systems[model_name] = new_rag
            training_progress['percentage'] = 70
            training_progress['status'] = 'Training RAG model...'
            
            # Clean up upload directory but keep processed texts
            processed_texts_backup = os.path.join(model_dir, "processed_texts")
            if os.path.exists(processed_texts_backup):
                shutil.rmtree(processed_texts_backup)
            shutil.copytree(processed_dir, processed_texts_backup)
            
            # 更新进度
            training_progress['percentage'] = 100
            training_progress['status'] = 'Training complete!'
            training_progress['is_training'] = False
            
            return jsonify({
                'success': True,
                'model_name': model_name,
                'processed_files': file_count
            })
        except Exception as e:
            logger.error(f"RAG training error: {str(e)}")
            training_progress['error'] = f"RAG training failed: {str(e)}"
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': f"RAG training failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_folder_for_rag: {str(e)}")
        training_progress['error'] = str(e)
        training_progress['percentage'] = 100
        training_progress['is_training'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/rename_rag_model', methods=['POST'])
def rename_rag_model():
    """Rename a RAG model."""
    try:
        data = request.get_json()
        old_name = data.get('old_name', '').strip()
        new_name = data.get('new_name', '').strip()
        
        if not old_name or not new_name:
            return jsonify({'error': 'Both old_name and new_name are required'}), 400
        
        # Sanitize new name - 允许中文字符、字母、数字和常用符号
        # 只过滤掉文件系统不允许的字符
        import string
        forbidden_chars = '<>:"/\\|?*'  # Windows文件系统禁止的字符
        sanitized_new_name = new_name
        for char in forbidden_chars:
            sanitized_new_name = sanitized_new_name.replace(char, '_')
        
        # 移除首尾空格并确保不为空
        sanitized_new_name = sanitized_new_name.strip()
        if not sanitized_new_name:
            return jsonify({'error': 'Invalid model name after sanitization'}), 400
        
        new_name = sanitized_new_name
        
        if old_name == new_name:
            return jsonify({'error': 'New name is the same as old name'}), 400
            
        # Check if old model exists
        if old_name not in rag_systems:
            return jsonify({'error': f'Model "{old_name}" not found'}), 404
            
        # Check if new name already exists
        if new_name in rag_systems:
            return jsonify({'error': f'Model "{new_name}" already exists'}), 409
            
        # Get old model directory
        old_model_dir = os.path.join(BASE_MODEL_DIR, old_name)
        new_model_dir = os.path.join(BASE_MODEL_DIR, new_name)
        
        if not os.path.exists(old_model_dir):
            return jsonify({'error': f'Model directory for "{old_name}" not found'}), 404
            
        # Rename directory
        try:
            os.rename(old_model_dir, new_model_dir)
        except OSError as e:
            logger.error(f"Failed to rename directory from {old_model_dir} to {new_model_dir}: {e}")
            return jsonify({'error': f'Failed to rename model directory: {str(e)}'}), 500
            
        # Update in-memory model registry
        rag_systems[new_name] = rag_systems.pop(old_name)
        
        # Update the model's save directory
        rag_systems[new_name].model_save_dir = Path(new_model_dir)
        
        logger.info(f"Successfully renamed RAG model from '{old_name}' to '{new_name}'")
        
        return jsonify({
            'success': True,
            'old_name': old_name,
            'new_name': new_name,
            'message': f'Model renamed from "{old_name}" to "{new_name}" successfully'
        })
        
    except Exception as e:
        logger.error(f"Error renaming RAG model: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/delete_rag_model', methods=['POST'])
def delete_rag_model():
    """Delete a RAG model."""
    try:
        data = request.get_json()
        model_name = data.get('model_name', '').strip()
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Prevent deletion of default model
        if model_name == 'default':
            return jsonify({'error': 'Cannot delete the default model'}), 400
            
        # Check if model exists
        if model_name not in rag_systems:
            return jsonify({'error': f'Model "{model_name}" not found'}), 404
            
        # Get model directory
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)
        
        if not os.path.exists(model_dir):
            return jsonify({'error': f'Model directory for "{model_name}" not found'}), 404
            
        # Remove from in-memory registry first
        rag_systems.pop(model_name, None)
        
        # Remove directory
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            logger.error(f"Failed to delete directory {model_dir}: {e}")
            return jsonify({'error': f'Failed to delete model directory: {str(e)}'}), 500
            
        logger.info(f"Successfully deleted RAG model '{model_name}'")
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'message': f'Model "{model_name}" deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting RAG model: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

@app.route('/clear-context', methods=['POST'])
def clear_context():
    """Clear conversation context for current session"""
    try:
        # Try to get session_id from JSON body first
        session_id = None
        if request.json and 'session_id' in request.json:
            session_id = request.json['session_id']
        elif 'session_id' in session:
            session_id = session['session_id']
        else:
            # Create new session if none exists
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
        # Clear the context for this specific session
        if session_id in conversation_contexts:
            conversation_contexts[session_id].clear()
            logger.info(f"Cleared context for session: {session_id}")
            message = f'Conversation context cleared for session {session_id[:8]}...'
        else:
            logger.info(f"No context found for session: {session_id}")
            message = 'No conversation context found to clear'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get-context', methods=['GET'])
def get_context():
    """Get conversation context for current session"""
    try:
        session_id = get_or_create_session_id()
        context = list(conversation_contexts.get(session_id, []))
        
        logger.info(f"Getting context for session_id: {session_id}, found {len(context)} messages")
        
        return jsonify({
            'status': 'success',
            'context': context,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

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
from flask import Response
import time
from pathlib import Path
import uuid

# Import document processor
from document_extractor import ChineseDocumentProcessor

# Import RAG system - adding the RAG directory to the path
rag_path = r'main_function'
sys.path.append(rag_path)
try:
    # Import directly from rag_trainer.py
    from rag_trainer import ChineseRAGSystem
    
    # RAG models directory
    BASE_MODEL_DIR = r"rag_models"

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
app.secret_key = 'your-secret-key-change-this-in-production'

# Store conversation contexts for each session
conversation_contexts = defaultdict(lambda: deque(maxlen=20))  # Keep last 20 messages 

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def format_markdown_to_user_friendly(text):
    """Convert markdown text to user-friendly plain text format."""
    if not text:
        return text
    
    # Store original text for comparison
    original_text = text
    
    # Remove markdown headers and replace with clean formatting
    text = re.sub(r'^###\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*40, text, flags=re.MULTILINE)
    text = re.sub(r'^##\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*50, text, flags=re.MULTILINE)
    text = re.sub(r'^#\s*\*\*(.*?)\*\*\s*$', r'\n\1\n' + '='*60, text, flags=re.MULTILINE)
    
    # Handle regular headers without bold
    text = re.sub(r'^###\s+(.*?)$', r'\n\1\n' + '-'*30, text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'\n\1\n' + '-'*40, text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'\n\1\n' + '-'*50, text, flags=re.MULTILINE)
    
    # Remove bold formatting but keep the content
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove italic formatting but keep the content
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Format numbered lists more clearly with proper spacing
    text = re.sub(r'^(\d+)\.\s+\*\*(.*?)\*\*\s*$', r'\n\1. \2', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\.\s+(.*?)$', r'\n\1. \2', text, flags=re.MULTILINE)
    
    # Format bullet points with better spacing
    text = re.sub(r'^[\*\-\+]\s+\*\*(.*?)\*\*\s*$', r'\n• \1', text, flags=re.MULTILINE)
    text = re.sub(r'^[\*\-\+]\s+(.*?)$', r'\n• \1', text, flags=re.MULTILINE)
    
    # Remove code blocks backticks but keep the content with better formatting
    text = re.sub(r'```.*?\n(.*?)```', r'\n[\1]\n', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'[\1]', text)
    
    # Handle special formatting patterns
    text = re.sub(r'\*\*Steps:\*\*', 'Steps:', text)
    text = re.sub(r'\*\*Note:\*\*', 'Note:', text)
    text = re.sub(r'\*\*Important:\*\*', 'Important:', text)
    
    # Clean up multiple newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove extra spaces while preserving single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Clean up lines and maintain readable structure
    lines = text.split('\n')
    formatted_lines = []
    prev_was_empty = False
    
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(line)
            prev_was_empty = False
        elif not prev_was_empty and formatted_lines:  # Keep single empty lines between content
            formatted_lines.append('')
            prev_was_empty = True
    
    # Join lines and clean up
    formatted_text = '\n'.join(formatted_lines).strip()
    
    # If formatting didn't improve readability much, return original with minimal changes
    if len(formatted_text) < len(original_text) * 0.5:
        # Just remove the most aggressive markdown and return
        simple_format = re.sub(r'\*\*(.*?)\*\*', r'\1', original_text)
        simple_format = re.sub(r'`([^`]+)`', r'\1', simple_format)
        return simple_format
    
    return formatted_text


# LM Studio API endpoint
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 日志用户问题
def log_user_question(question, session_id=None):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"{timestamp} | session: {session_id or '-'} | {question}\n"
        log_path = os.path.join(os.path.dirname(__file__), 'user_question_log.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
    except Exception as e:
        logger.error(f"Failed to log user question: {e}")

# Add a helper function to call the LLM API with proper error handling
def call_llm_api(messages, temperature=0.5, max_tokens=32000, stream=False):
    """Helper function to call the LLM API with consistent error handling"""
    try:
        payload = {
            "model": "qwen/qwen3-14b", 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        logger.info(f"Sending request to LLM API: {LM_STUDIO_API_URL}")
        logger.debug(f"Payload: {payload}")
        
        response = requests.post(
            LM_STUDIO_API_URL, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=300,
            stream=stream
        )
        
        logger.info(f"LLM API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None, f"LLM API error: {response.text}"
        
        if stream:
            return response, None
        else:
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

def build_conversation_context(session_id, new_message, system_prompt=None):
    """Build conversation context with history"""
    context = conversation_contexts[session_id]
    
    # Add system message if provided and context is empty
    messages = []
    if system_prompt and (not context or context[0].get('role') != 'system'):
        messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    messages.extend(list(context))
    
    # Add new user message
    messages.append({"role": "user", "content": new_message})
    
    return messages

def update_conversation_context(session_id, user_message, ai_response):
    """Update conversation context with new messages"""
    context = conversation_contexts[session_id]
    context.append({"role": "user", "content": user_message})
    context.append({"role": "assistant", "content": ai_response})

def get_or_create_session_id():
    """Get session ID from request or create new one"""
    # First try to get from request JSON data (only for POST/PUT requests with JSON content)
    if request.method in ['POST', 'PUT'] and request.is_json and request.json and 'session_id' in request.json:
        return request.json['session_id']
    
    # Then try to get from URL parameters (for GET requests)  
    if 'session_id' in request.args:
        return request.args['session_id']
    
    # Finally try to get from Flask session or create new one
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


# Remove old individual extraction functions and replace with unified processor
def extract_file_content(file_path, file_extension):
    """Extract content using the unified document processor."""
    if not os.path.exists(file_path):
        return "Error: File not found"
    
    try:
        # Create a temporary directory for processing
        temp_dir = os.path.dirname(file_path)
        processor = ChineseDocumentProcessor(temp_dir)
        
        # Process the specific file based on extension
        if file_extension == '.docx':
            content = processor.extract_text_from_docx(file_path)
        elif file_extension == '.pdf':
            content = processor.extract_text_from_pdf(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            content = processor.extract_text_from_excel(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        else:
            return "Error: Unsupported file format"
        
        return content if content else "Error: No content extracted"
        
    except Exception as e:
        logger.error(f"Error in extract_file_content: {str(e)}")
        return f"Error extracting file content: {str(e)}"

# 存储当前训练进度的全局变量
training_progress = {
    'percentage': 0,
    'status': '',
    'is_training': False,
    'error': None
}

# SSE 路由，用于实时推送训练进度
@app.route('/training_progress', methods=['GET'])
def training_progress_stream():
    def generate():
        while True:
            yield f"data: {json.dumps(training_progress)}\n\n"
            time.sleep(1)  # 每秒推送一次
    return Response(generate(), mimetype='text/event-stream')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle text-only questions with context (non-streaming)."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400


        # Get or create session ID
        session_id = get_or_create_session_id()

        # 日志用户问题
        log_user_question(question, session_id)

        # System prompt for tool usage
        system_prompt = (
            "你是一个友善的助手。你可以参考之前的对话内容来提供更好的回答。"
            "请用清晰、简洁的语言回答，避免过多的标记符号。"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        # Build conversation context with history
        messages = build_conversation_context(session_id, question, system_prompt)

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
        
        # Format the response to be more user-friendly
        response_text = format_markdown_to_user_friendly(response_text)

        # Update conversation context - 只保存处理后的回答，不包含思考内容
        update_conversation_context(session_id, question, response_text)

        result = {'thinking': thinking, 'response': response_text, 'session_id': session_id}
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask-stream', methods=['POST'])
def ask_question_stream():
    """Handle streaming questions with context"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400


        # Get or create session ID
        session_id = get_or_create_session_id()

        # 日志用户问题
        log_user_question(question, session_id)

        # System prompt for tool usage
        system_prompt = (
            "你是一个友善的助手。你可以参考之前的对话内容来提供更好的回答。"
            "请用清晰、简洁的语言回答，避免过多的标记符号。"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        # Build conversation context with history
        messages = build_conversation_context(session_id, question, system_prompt)

        def generate():
            ai_response_content = ""
            try:
                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        # Send raw content for streaming (format at end)
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the complete response before saving
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

    except Exception as e:
        logger.error(f"Error in /ask-stream: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload_and_ask', methods=['POST'])
def upload_and_ask():
    """Handle questions with optional multiple file uploads using streaming response and context memory."""
    try:
        # Debug logging for file upload
        logger.debug(f"request.files keys: {list(request.files.keys())}")
        logger.debug(f"request.files.getlist('file') length: {len(request.files.getlist('file'))}")
        logger.debug(f"request.form: {request.form}")
        question = request.form.get('question', '')
        files = request.files.getlist('file')
        file_contents = []


        # Get or create session ID
        session_id = request.form.get('session_id') or str(uuid.uuid4())

        # 日志用户问题
        log_user_question(question, session_id)

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

        # System prompt for file processing
        system_prompt = (
            "你是一个友善的助手，你必须基于提供的信息来回答问题。你可以参考之前的对话内容来提供更好的回答。"
            "请用清晰、简洁的语言回答，避免过多的标记符号。"
            "在回答之前，请在 <think> 标签中提供你的推理。"
        )

        # Build conversation context with history
        messages = build_conversation_context(session_id, combined_input, system_prompt)

        def generate():
            ai_response_content = ""
            try:
                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the response for better user experience
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)  # 只保存原始问题，不包含文件内容
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

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
    """Handle questions using RAG system with streaming response and context memory."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        model_id = data.get('model_id', 'default') 
        session_id = data.get('session_id') or str(uuid.uuid4())
        

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # 日志用户问题
        log_user_question(question, session_id)

        if not RAG_AVAILABLE or not rag_systems:
            return jsonify({'error': 'RAG system is not available'}), 503

        # Check if the requested model exists
        if model_id not in rag_systems:
            return jsonify({'error': f"RAG model '{model_id}' not found"}), 404

        # Log that we're using RAG
        logger.info(f"Using RAG system '{model_id}' for question: {question}")
        
        def generate():
            ai_response_content = ""
            try:
                # Get relevant context from RAG system
                try:
                    # Use RAG system to get context and generate response
                    rag_context = rag_systems[model_id].retrieve_relevant_docs(question)
                    
                    # Build enhanced prompt with RAG context and conversation history
                    system_prompt = (
                        f"你是一个友善的助手，基于提供的知识库内容和之前的对话历史来回答问题。"
                        f"请用清晰、简洁的语言回答，避免过多的标记符号。"
                        f"在回答之前，请在 <think> 标签中提供你的推理。\n\n"
                        f"相关知识库内容：\n{rag_context}"
                    )
                    
                    # Build conversation context with history
                    messages = build_conversation_context(session_id, question, system_prompt)
                    
                except Exception as rag_error:
                    logger.error(f"RAG retrieval error: {str(rag_error)}")
                    yield f"data: {json.dumps({'error': f'RAG retrieval error: {str(rag_error)}'})}\n\n"
                    return

                # Call the LLM API with streaming enabled
                response, error = call_llm_api(messages, stream=True)
                
                if error:
                    yield f"data: {json.dumps({'error': error})}\n\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        
                        # Skip empty lines and comments
                        if not line.strip() or line.startswith('#'):
                            continue
                            
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                        
                        # Skip [DONE] signal
                        if line.strip() == '[DONE]':
                            break
                            
                        try:
                            # Parse the JSON chunk
                            chunk_data = json.loads(line)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        ai_response_content += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                
                # Update conversation context with complete response
                if ai_response_content:
                    # Process thinking and response parts - 分离思考内容和实际回答
                    think_match = re.search(r'<think>(.*?)</think>', ai_response_content, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    response_text = ai_response_content[think_match.end():].strip() if think_match else ai_response_content
                    # Format the response for better user experience
                    formatted_response_text = format_markdown_to_user_friendly(response_text)
                    # 只保存处理后的回答内容，不包含思考标签
                    update_conversation_context(session_id, question, formatted_response_text)
                    
                    # Send the final formatted response with thinking content
                    yield f"data: {json.dumps({'formatted_response': formatted_response_text, 'thinking': thinking_content})}\n\n"
                            
                # Send end signal with session info
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in RAG streaming: {str(e)}")
                yield f"data: {json.dumps({'error': f'RAG streaming error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
            
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
    """Process files in a directory for RAG training using unified document processor."""
    processed_dir = os.path.join(upload_dir, "processed_texts")
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # 尝试检测和修复PDF文件
        for root, dirs, files in os.walk(upload_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    try:
                        # 尝试打开PDF文件验证其完整性
                        try:
                            with open(file_path, 'rb') as f:
                                # 尝试读取前1024字节检查PDF头
                                header = f.read(1024)
                                if not header.startswith(b'%PDF-'):
                                    logger.warning(f"File doesn't have PDF header: {file_path}")
                                    # 如果不是PDF格式，尝试重命名为txt文件
                                    txt_path = file_path.replace('.pdf', '.txt')
                                    os.rename(file_path, txt_path)
                                    logger.info(f"Renamed problematic PDF to text file: {txt_path}")
                        except Exception as e:
                            logger.error(f"Error checking PDF file {file_path}: {e}")
                    except Exception as check_error:
                        logger.error(f"Error during PDF check: {check_error}")

        # 使用统一文档处理器
        processor = ChineseDocumentProcessor(upload_dir)
        
        # 设置尝试次数来处理有问题的PDF
        processor.pdf_retry_attempts = 3
        processor.robust_pdf_handling = True
        
        processed_texts = processor.process_documents()
        
        # 保存处理后的文本
        file_count = 0
        for file_path, text in processed_texts.items():
            if text.strip():  # 只保存非空文本
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                processed_file = os.path.join(processed_dir, f"{base_name}_processed.txt")
                
                with open(processed_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                file_count += 1
                logger.info(f"Processed and saved: {processed_file}")
        
        return processed_dir, file_count
        
    except Exception as e:
        logger.error(f"Error in process_folder_for_rag: {e}")
        return processed_dir, 0

# Add a route to handle folder upload and RAG training
# 修改 upload_folder_for_rag 路由以更新训练进度
@app.route('/upload_folder_for_rag', methods=['POST'])
def upload_folder_for_rag():
    """Handle folder upload and train RAG model."""
    global training_progress
    try:
        # 初始化训练进度
        training_progress = {
            'percentage': 10,
            'status': 'Uploading files...',
            'is_training': True,
            'error': None
        }

        # Check if files were uploaded
        if 'files[]' not in request.files:
            training_progress['error'] = 'No files provided'
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': 'No files provided'}), 400
            
        # Get folder name from form data (instead of user input)
        folder_name = request.form.get('folder_name', '')
        if not folder_name:
            # 如果没有提供文件夹名称，生成一个默认名称
            folder_name = f"rag_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sanitize folder name
        model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
        
        # Check if model name already exists, add suffix if needed
        original_model_name = model_name
        counter = 1
        while model_name in rag_systems:
            model_name = f"{original_model_name}_{counter}"
            counter += 1
        
        # Create directory for this upload
        upload_dir = os.path.join('uploads', f"rag_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save all uploaded files
        files = request.files.getlist('files[]')
        saved_paths = []
        
        for file in files:
            if file.filename == '':
                continue
                
            filepath_parts = file.filename.split('/')
            if len(filepath_parts) > 1:
                subdir = os.path.join(upload_dir, *filepath_parts[:-1])
                os.makedirs(subdir, exist_ok=True)
                save_path = os.path.join(upload_dir, *filepath_parts)
            else:
                save_path = os.path.join(upload_dir, file.filename)
                
            file.save(save_path)
            saved_paths.append(save_path)
            
        logger.info(f"Saved {len(saved_paths)} files for RAG training")
        
        # 检查文件类型并记录PDF文件
        pdf_files = []
        for path in saved_paths:
            if path.lower().endswith('.pdf'):
                pdf_files.append(path)
        
        # 更新进度
        training_progress['percentage'] = 10
        training_progress['status'] = f'Processing {len(saved_paths)} files (including {len(pdf_files)} PDFs)...'

        # 处理文件
        processed_dir, file_count = process_folder_for_rag(upload_dir)
        
        if file_count == 0:
            error_msg = 'No valid files for processing. PDF files might be corrupted or in an unsupported format.'
            training_progress['error'] = error_msg
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': error_msg}), 400
            
        # 更新进度
        training_progress['percentage'] = 30
        training_progress['status'] = f'Successfully processed {file_count} files. Training RAG model...'

        # Create model directory
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Train RAG model
        try:
            new_rag = ChineseRAGSystem(processed_texts_dir=processed_dir, model_save_dir=model_dir)
            new_rag.train_system(processed_dir)
            training_progress['percentage'] = 50
            training_progress['status'] = 'Training RAG model...'
            
            # Store the new model
            rag_systems[model_name] = new_rag
            training_progress['percentage'] = 70
            training_progress['status'] = 'Training RAG model...'
            
            # Clean up upload directory but keep processed texts
            processed_texts_backup = os.path.join(model_dir, "processed_texts")
            if os.path.exists(processed_texts_backup):
                shutil.rmtree(processed_texts_backup)
            shutil.copytree(processed_dir, processed_texts_backup)
            
            # 更新进度
            training_progress['percentage'] = 100
            training_progress['status'] = 'Training complete!'
            training_progress['is_training'] = False
            
            return jsonify({
                'success': True,
                'model_name': model_name,
                'processed_files': file_count
            })
        except Exception as e:
            logger.error(f"RAG training error: {str(e)}")
            training_progress['error'] = f"RAG training failed: {str(e)}"
            training_progress['percentage'] = 100
            training_progress['is_training'] = False
            return jsonify({'error': f"RAG training failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_folder_for_rag: {str(e)}")
        training_progress['error'] = str(e)
        training_progress['percentage'] = 100
        training_progress['is_training'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/rename_rag_model', methods=['POST'])
def rename_rag_model():
    """Rename a RAG model."""
    try:
        data = request.get_json()
        old_name = data.get('old_name', '').strip()
        new_name = data.get('new_name', '').strip()
        
        if not old_name or not new_name:
            return jsonify({'error': 'Both old_name and new_name are required'}), 400
        
        # Sanitize new name
        new_name = re.sub(r'[^a-zA-Z0-9_-]', '_', new_name)
        
        if old_name == new_name:
            return jsonify({'error': 'New name is the same as old name'}), 400
            
        # Check if old model exists
        if old_name not in rag_systems:
            return jsonify({'error': f'Model "{old_name}" not found'}), 404
            
        # Check if new name already exists
        if new_name in rag_systems:
            return jsonify({'error': f'Model "{new_name}" already exists'}), 409
            
        # Get old model directory
        old_model_dir = os.path.join(BASE_MODEL_DIR, old_name)
        new_model_dir = os.path.join(BASE_MODEL_DIR, new_name)
        
        if not os.path.exists(old_model_dir):
            return jsonify({'error': f'Model directory for "{old_name}" not found'}), 404
            
        # Rename directory
        try:
            os.rename(old_model_dir, new_model_dir)
        except OSError as e:
            logger.error(f"Failed to rename directory from {old_model_dir} to {new_model_dir}: {e}")
            return jsonify({'error': f'Failed to rename model directory: {str(e)}'}), 500
            
        # Update in-memory model registry
        rag_systems[new_name] = rag_systems.pop(old_name)
        
        # Update the model's save directory
        rag_systems[new_name].model_save_dir = Path(new_model_dir)
        
        logger.info(f"Successfully renamed RAG model from '{old_name}' to '{new_name}'")
        
        return jsonify({
            'success': True,
            'old_name': old_name,
            'new_name': new_name,
            'message': f'Model renamed from "{old_name}" to "{new_name}" successfully'
        })
        
    except Exception as e:
        logger.error(f"Error renaming RAG model: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/delete_rag_model', methods=['POST'])
def delete_rag_model():
    """Delete a RAG model."""
    try:
        data = request.get_json()
        model_name = data.get('model_name', '').strip()
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Prevent deletion of default model
        if model_name == 'default':
            return jsonify({'error': 'Cannot delete the default model'}), 400
            
        # Check if model exists
        if model_name not in rag_systems:
            return jsonify({'error': f'Model "{model_name}" not found'}), 404
            
        # Get model directory
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)
        
        if not os.path.exists(model_dir):
            return jsonify({'error': f'Model directory for "{model_name}" not found'}), 404
            
        # Remove from in-memory registry first
        rag_systems.pop(model_name, None)
        
        # Remove directory
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            logger.error(f"Failed to delete directory {model_dir}: {e}")
            return jsonify({'error': f'Failed to delete model directory: {str(e)}'}), 500
            
        logger.info(f"Successfully deleted RAG model '{model_name}'")
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'message': f'Model "{model_name}" deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting RAG model: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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

@app.route('/clear-context', methods=['POST'])
def clear_context():
    """Clear conversation context for current session"""
    try:
        # Try to get session_id from JSON body first
        session_id = None
        if request.json and 'session_id' in request.json:
            session_id = request.json['session_id']
        elif 'session_id' in session:
            session_id = session['session_id']
        else:
            # Create new session if none exists
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
        # Clear the context for this specific session
        if session_id in conversation_contexts:
            conversation_contexts[session_id].clear()
            logger.info(f"Cleared context for session: {session_id}")
            message = f'Conversation context cleared for session {session_id[:8]}...'
        else:
            logger.info(f"No context found for session: {session_id}")
            message = 'No conversation context found to clear'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get-context', methods=['GET'])
def get_context():
    """Get conversation context for current session"""
    try:
        session_id = get_or_create_session_id()
        context = list(conversation_contexts.get(session_id, []))
        
        return jsonify({
            'status': 'success',
            'context': context,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

