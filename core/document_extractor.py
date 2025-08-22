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
                        content = self.extract_text_from_docx_to_markdown(file_path)
                        processed_texts[file_path] = self.clean_markdown_text(content)
                    
                    elif file_extension == '.pdf':
                        logger.info(f"Process documents: {file_path}")
                        content = self.extract_text_from_pdf_to_markdown(file_path)
                        processed_texts[file_path] = self.clean_markdown_text(content)
                    
                    elif file_extension in ['.xlsx', '.xls']:
                        logger.info(f"Process documents: {file_path}")
                        content = self.extract_text_from_excel_to_markdown(file_path)
                        processed_texts[file_path] = self.clean_markdown_text(content)
                    
                    elif file_extension == '.txt':
                        logger.info(f"Process documents: {file_path}")
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        processed_texts[file_path] = self.clean_markdown_text(content)
                    
                    elif file_extension == '.md':
                        logger.info(f"Process documents: {file_path}")
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        processed_texts[file_path] = self.clean_markdown_text(content)
                except Exception as e:
                    logger.error(f"Fail to extract {file_extension} files from: {file_path}: {e}")
                    processed_texts[file_path] = f"[提取失败: {os.path.basename(file_path)}]"
        
        return processed_texts
    
    

    def clean_markdown_text(self, text: str) -> str:
        """Clean text while preserving Markdown structure for RAG."""
        if not text or not text.strip():
            logger.warning("Input text is empty or whitespace-only")
            return ""
        
        try:
            # Remove control characters but preserve markdown syntax
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Remove specific patterns but keep markdown formatting
            text = re.sub(r'\b0x[0-9a-fA-F]{4,}\b', '', text)
            text = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '', text)
            text = re.sub(r'\b(Error|Warning|Exception|Failed):?\s*[A-Za-z0-9_-]+\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
            
            # Clean up excessive whitespace but preserve markdown structure
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
            text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
            
            # Remove lines that are too short but preserve headers and lists
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    filtered_lines.append('')
                    continue
                
                # Keep markdown headers
                if line.startswith('#'):
                    filtered_lines.append(line)
                # Keep markdown lists
                elif line.startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.\s', line):
                    filtered_lines.append(line)
                # Keep table rows
                elif '|' in line:
                    filtered_lines.append(line)
                # Keep other content if it's long enough
                elif len(line) > 10 or re.search(r'[\u4e00-\u9fff]', line):
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines).strip()
            
        except Exception as e:
            logger.error(f"Error cleaning markdown text: {e}")
            return text.strip()

    def extract_text_from_docx_to_markdown(self, file_path: str) -> str:
        """Extract text from DOCX and convert to Markdown format."""
        try:
            doc = DocxDocument(file_path)
            markdown_lines = []
            
            # Add document title
            filename = os.path.splitext(os.path.basename(file_path))[0]
            markdown_lines.append(f"# {filename}")
            markdown_lines.append("")
            
            current_header_level = 1
            
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                    
                text = para.text.strip()
                
                # Try to detect headers based on text characteristics
                if self._is_likely_header(text):
                    current_header_level = min(current_header_level + 1, 4)
                    markdown_lines.append(f"{'#' * current_header_level} {text}")
                    markdown_lines.append("")
                else:
                    # Regular paragraph
                    markdown_lines.append(text)
                    markdown_lines.append("")
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.error(f"Error extracting DOCX to markdown from {file_path}: {e}")
            return ""

    def extract_text_from_pdf_to_markdown(self, file_path: str) -> str:
        """Extract text from PDF and convert to Markdown format."""
        try:
            doc = fitz.open(file_path)
            markdown_lines = []
            
            # Add document title
            filename = os.path.splitext(os.path.basename(file_path))[0]
            markdown_lines.append(f"# {filename}")
            markdown_lines.append("")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    # Add page separator for multi-page documents
                    if page_num > 0:
                        markdown_lines.append(f"## Page {page_num + 1}")
                        markdown_lines.append("")
                    
                    # Process text line by line
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Try to detect headers
                        if self._is_likely_header(line):
                            markdown_lines.append(f"### {line}")
                            markdown_lines.append("")
                        else:
                            markdown_lines.append(line)
                    
                    markdown_lines.append("")
            
            doc.close()
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.error(f"Failed to extract PDF to markdown {file_path}: {e}")
            return f"# {os.path.basename(file_path)}\n\n[PDF提取失败: {str(e)}]"

    def extract_text_from_excel_to_markdown(self, file_path: str) -> str:
        """Extract text from Excel and convert to Markdown format."""
        try:
            wb = load_workbook(file_path, read_only=True, data_only=True)
            markdown_lines = []
            
            # Add document title
            filename = os.path.splitext(os.path.basename(file_path))[0]
            markdown_lines.append(f"# {filename}")
            markdown_lines.append("")
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                
                # Add sheet as a section
                markdown_lines.append(f"## {sheet_name}")
                markdown_lines.append("")
                
                # Get all data from the sheet
                data = []
                for row in ws.rows:
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value))
                        else:
                            row_values.append("")
                    data.append(row_values)
                
                # Filter out empty rows
                data = [row for row in data if any(cell.strip() for cell in row)]
                
                if data:
                    # Create markdown table
                    if len(data) > 1:
                        # Header row
                        header = "| " + " | ".join(data[0]) + " |"
                        separator = "|" + "|".join([" --- " for _ in data[0]]) + "|"
                        markdown_lines.append(header)
                        markdown_lines.append(separator)
                        
                        # Data rows
                        for row in data[1:]:
                            row_md = "| " + " | ".join(row) + " |"
                            markdown_lines.append(row_md)
                    else:
                        # Single row, treat as list
                        for item in data[0]:
                            if item.strip():
                                markdown_lines.append(f"- {item}")
                
                markdown_lines.append("")
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.error(f"Error extracting Excel to markdown from {file_path}: {e}")
            return f"# {os.path.basename(file_path)}\n\n[Excel提取失败: {str(e)}]"

    def _is_likely_header(self, text: str) -> bool:
        """Determine if a line of text is likely a header."""
        if len(text) > 100:  # Too long to be a header
            return False
        
        # Check for common header patterns
        header_patterns = [
            r'^[A-Z\u4e00-\u9fff][A-Z\u4e00-\u9fff\s]{2,50}$',  # All caps or Chinese
            r'^\d+[\.\)]\s*[A-Za-z\u4e00-\u9fff]',  # Numbered sections
            r'^第[一二三四五六七八九十\d]+[章节部分条款]',  # Chinese section markers
            r'^[A-Za-z\u4e00-\u9fff][^\.]{5,50}$',  # Short lines without periods
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text):
                return True
                
        return False

# if __name__ == "__main__":
#     processor = ChineseDocumentProcessor(output_dir="processed_texts")
    # results = processor.process_documents()
    # print(f"Processed {len(results)} documents")
    # for path, content in results.items():
    #     print(f"{path}: {len(content)} characters")
