import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import re
import fitz 
from docx import Document as DocxDocument
from openpyxl import load_workbook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseDocumentProcessor:
    """Handle document processing for Chinese and English texts."""

    def __init__(self, output_dir: str = "processed_texts"):
        """Initialize document processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and optimize text for RAG embedding model, designed for Chinese and English documents."""
        if not text or not text.strip():
            # Debug Message
            logger.warning("Input text is empty or whitespace-only")
            return ""
        
        try:
            # Remove control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            # Remove common garbled text patterns (keep Chinese, English, numbers, and common punctuation)
            text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s.,!?;:\-()（）《》【】“”‘’、，。！？；：]+', ' ', text)

            # Remove specific patterns (e.g., hex codes, UUIDs, error messages, PDF headers/footers and duplicate punctuation)
            text = re.sub(r'\b0x[0-9a-fA-F]{4,}\b', '', text)
            text = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '', text)
            text = re.sub(r'\b(Error|Warning|Exception|Failed):?\s*[A-Za-z0-9_-]+\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'([.,!?;:\-])\1+', r'\1', text)

            # Remove duplicate whitespace
            text = re.sub(r'\s+', ' ', text) 
            text = re.sub(r'\n+', '\n', text)
            
            # Remove leading and trailing whitespace
            text = text.strip()

            # Process unordered text, attempt to reorganize by sentence
            sentences = []
            current_sentence = []
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Use defined sentence splitting
                parts = re.split(r'([。！？.!?])', line)
                for i, part in enumerate(parts):
                    if part in '。！？.!?':
                        if current_sentence:
                            current_sentence.append(part)
                            sentences.append(''.join(current_sentence))
                            current_sentence = []
                    else:
                        if part.strip():
                            current_sentence.append(part)

            # Add remaining sentence
            if current_sentence:
                sentences.append(''.join(current_sentence))

            # Remove too short sentences
            filtered_sentences = []
            has_chinese = any(re.search(r'[\u4e00-\u9fff]', s) for s in sentences)
            sentence_lengths = [len(s.strip()) for s in sentences]
            for s in sentences:
                # Keep Chinese sentences (length > 3) or English sentences (length > 6)
                if has_chinese and re.search(r'[\u4e00-\u9fff]', s) and len(s.strip()) > 3:
                    filtered_sentences.append(s)
                elif not has_chinese and len(s.strip()) > 6:
                    filtered_sentences.append(s)

            # Check cleaned text quality
            cleaned_text = '\n'.join(filtered_sentences)
            if not cleaned_text.strip():
                logger.warning(
                    f"Text is empty after filtering. Has Chinese: {has_chinese}, "
                    f"Sentence count: {len(sentences)}, Kept: {len(filtered_sentences)}, "
                    f"Sentence lengths: {sentence_lengths[:10]}..."
                )
                # Fallback: Return minimally cleaned text
                minimal_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
                minimal_text = re.sub(r'\b0x[0-9a-fA-F]{4,}\b', '', minimal_text)
                minimal_text = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '', minimal_text)
                minimal_text = re.sub(r'\b(Error|Warning|Exception|Failed):?\s*[A-Za-z0-9_-]+\b', '', minimal_text, flags=re.IGNORECASE)
                minimal_text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', minimal_text, flags=re.IGNORECASE)
                minimal_text = re.sub(r'([.,!?;:\-])\1+', r'\1', minimal_text)
                minimal_text = re.sub(r'\s+', ' ', minimal_text).strip()
                if minimal_text:
                    logger.info("Returning minimally cleaned text to avoid empty output")
                    return minimal_text
                logger.warning("Minimal cleaned text is also empty")
                return ""

            # Remove isolated punctuations
            cleaned_text = re.sub(r'(?<![\u4e00-\u9fff])[.,;:!?](?![\u4e00-\u9fff])\s*', ' ', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
            return cleaned_text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text.strip() 
    
    def process_documents(self) -> Dict[str, str]:
        """Process all documents in the specified directory and clean extracted text."""
        processed_texts = {}
        
        # Debug Message
        logger.info(f"Processing documents in {self.output_dir}")
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                
                try:
                    if file_extension == '.docx':
                        logger.info(f"Process documents: {file_path}")
                        content = self.extract_text_from_docx(file_path)
                        processed_texts[file_path] = self.clean_text(content)
                    
                    elif file_extension == '.pdf':
                        logger.info(f"Process documents: {file_path}")
                        content = self.extract_text_from_pdf(file_path)
                        processed_texts[file_path] = self.clean_text(content)
                    
                    elif file_extension in ['.xlsx', '.xls']:
                        logger.info(f"Process documents: {file_path}")
                        content = self.extract_text_from_excel(file_path)
                        processed_texts[file_path] = self.clean_text(content)
                    
                    elif file_extension == '.txt':
                        logger.info(f"Process documents: {file_path}")
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        processed_texts[file_path] = self.clean_text(content)
                except Exception as e:
                    logger.error(f"Fail to extract {file_extension} files from: {file_path}: {e}")
                    processed_texts[file_path] = f"[提取失败: {os.path.basename(file_path)}]"
        
        return processed_texts
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = DocxDocument(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            logger.error(f"Error extracting DOCX content from {file_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                # Debug Message
                logger.info(f"Successfully extracted text from PDF using PyMuPDF: {file_path}")
                return text
            else:
                logger.warning(f"No text extracted from PDF: {file_path}")
                return f"[No text extracted from PDF: {os.path.basename(file_path)}]"
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            return f"[PDF提取失败: {os.path.basename(file_path)}]"
    
    def extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from an Excel file."""
        try:
            wb = load_workbook(file_path, read_only=True, data_only=True)
            text = []
            
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                sheet_text = [f"--- Sheet: {sheet} ---"]

                # read cell data
                for row in ws.rows:
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value))
                    if row_values:
                        sheet_text.append("\t".join(row_values))

                # If the sheet has content
                if len(sheet_text) > 1:
                    text.extend(sheet_text)
                    # Add a blank line to separate different sheets
                    text.append("")
            
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting Excel content from {file_path}: {e}")
            return ""

# if __name__ == "__main__":
#     processor = ChineseDocumentProcessor(output_dir="processed_texts")
    # results = processor.process_documents()
    # print(f"Processed {len(results)} documents")
    # for path, content in results.items():
    #     print(f"{path}: {len(content)} characters")
