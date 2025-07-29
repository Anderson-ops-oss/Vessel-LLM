# document_extractor.py

import logging
from pathlib import Path
import pdfplumber
from docx import Document
import pandas as pd
from typing import Dict

# setting the log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseDocumentProcessor:
    """handles processing of Chinese documents including PDF, DOCX, and Excel files."""
    
    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = Path(knowledge_dir)
        self.processed_texts = {}
        
    def clean_text(self, text: str) -> str:
        """clean and standardize text"""
        if not text:
            return ""
        # remove non-Chinese characters, keep ASCII and whitespace
        text = ''.join(c for c in text if '\u4e00' <= c <= '\u9fff' or c.isascii() or c.isspace())
        
        # standardize whitespace
        text = ' '.join(text.split())
        
        # replace common punctuation with standard ASCII equivalents
        text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """extract text from PDF files"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            logger.info(f"Successful extracted .pdf files from: {pdf_path}")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Fail to extracted .pdf files from: {pdf_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """extract text from DOCX files"""
        try:
            doc = Document(docx_path)
            text = '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
            logger.info(f"Successful extracted .docx files from: {docx_path}")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Fail to extracted .docx files from {docx_path}: {e}")
            return ""
    
    def extract_text_from_excel(self, excel_path: str) -> str:
        """extract text from Excel files"""
        try:
            xls = pd.ExcelFile(excel_path)
            texts = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                texts.append(' '.join(df.astype(str).stack().values))
            text = '\n'.join(texts)
            logger.info(f"Successful extracted .xls, .xlsx files from: {excel_path}")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Fail to extracted .xls, .xlsx files from: {excel_path}: {e}")
            return ""
    
    def process_documents(self) -> Dict[str, str]:
        """Process all documents in the knowledge directory and extract text."""
        supported_extensions = {'.pdf', '.docx', '.xlsx', '.xls'}
        for file_path in self.knowledge_dir.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                logger.info(f"Process documents: {file_path}")
                try:
                    if file_path.suffix.lower() == '.pdf':
                        text = self.extract_text_from_pdf(str(file_path))
                    elif file_path.suffix.lower() == '.docx':
                        text = self.extract_text_from_docx(str(file_path))
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        text = self.extract_text_from_excel(str(file_path))
                    else:
                        continue
                    if text:
                        self.processed_texts[str(file_path)] = text
                except Exception as e:
                    logger.error(f"Fail to process documents {file_path}: {e}")
        return self.processed_texts
    
    def save_texts(self, output_dir: str = "processed_texts"):
        """Save processed texts to the specified output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        for file_path, text in self.processed_texts.items():
            output_file = output_path / f"{Path(file_path).stem}_processed.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Save procerssed texts to: {output_file}")

# --- Execution Block for Document Extraction ---
if __name__ == "__main__":
    KNOWLEDGE_DIR = r"C:\Users\CHENGAN3\Desktop\Annual Report Test Case\Chinese"  
    PROCESSED_DIR = "processed_texts"

    print("Processing documents...")
    processor = ChineseDocumentProcessor(KNOWLEDGE_DIR)
    processor.process_documents()
    processor.save_texts(PROCESSED_DIR)
    print("Document processing complete!")