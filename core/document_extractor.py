import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import re
import fitz  # PyMuPDF as the primary PDF reader

# 文档处理库
from docx import Document as DocxDocument
from openpyxl import load_workbook

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseDocumentProcessor:
    """处理各种格式文档的类，提取并清理文本内容"""
    
    def __init__(self, output_dir: str = "processed_texts"):
        """初始化文档处理器
        
        Args:
            output_dir: 处理后文本的输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """清理文本，移除无关内容、乱码，并优化文本顺序以适合RAG嵌入模型，专为中英文文档设计
        
        Args:
            text: 原始提取的文本
            
        Returns:
            清理并优化后的文本
        """
        if not text or not text.strip():
            logger.warning("Input text is empty or whitespace-only")
            return ""
        
        try:
            # 1. 移除乱码和控制字符
            # 移除控制字符（\x00-\x08, \x0B, \x0C, \x0E-\x1F, \x7F），保留换行符(\n)
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            # 移除常见乱码模式（保留中文、英文、数字、常见标点）
            text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s.,!?;:\-()（）《》【】“”‘’、，。！？；：]+', ' ', text)
            
            # 2. 移除无关内容
            # 十六进制错误代码 (如 0x1234abcd)
            text = re.sub(r'\b0x[0-9a-fA-F]{4,}\b', '', text)
            # UUID (如 550e8400-e29b-41d4-a716-446655440000)
            text = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '', text)
            # 通用错误消息 (如 Error: 12345, Warning: ABC123)
            text = re.sub(r'\b(Error|Warning|Exception|Failed):?\s*[A-Za-z0-9_-]+\b', '', text, flags=re.IGNORECASE)
            # PDF页眉/页脚 (如 "Page 1 of 10")
            text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
            # 移除重复标点 (如 "....", "!!!")
            text = re.sub(r'([.,!?;:\-])\1+', r'\1', text)
            
            # 3. 清理冗余空格和换行
            text = re.sub(r'\s+', ' ', text)  # 多个空格/换行替换为单个空格
            text = re.sub(r'\n+', '\n', text)  # 多个换行替换为单个换行
            text = text.strip()  # 移除首尾空白
            
            # 4. 处理无序文本，尝试按句子重组
            sentences = []
            current_sentence = []
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 使用中英文句末标点（。！？.!?）分割句子
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
            
            # 添加未结束的句子
            if current_sentence:
                sentences.append(''.join(current_sentence))
            
            # 5. 过滤掉过短的句子（针对中英文优化）
            filtered_sentences = []
            has_chinese = any(re.search(r'[\u4e00-\u9fff]', s) for s in sentences)
            sentence_lengths = [len(s.strip()) for s in sentences]
            for s in sentences:
                # 保留中文句子（长度>3）或英文句子（长度>6）
                if has_chinese and re.search(r'[\u4e00-\u9fff]', s) and len(s.strip()) > 3:
                    filtered_sentences.append(s)
                elif not has_chinese and len(s.strip()) > 6:
                    filtered_sentences.append(s)
            
            # 6. 验证清理后的文本
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
            
            # 7. 最终清理孤立标点（保留中文标点）
            cleaned_text = re.sub(r'(?<![\u4e00-\u9fff])[.,;:!?](?![\u4e00-\u9fff])\s*', ' ', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            logger.info(f"Successfully cleaned text. Sentences kept: {len(filtered_sentences)}/{len(sentences)}, Has Chinese: {has_chinese}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text.strip()  # 返回原始文本以避免数据丢失
    
    def process_documents(self) -> Dict[str, str]:
        """处理指定目录下的所有文档，并清理提取的文本
        
        Returns:
            字典，键为文件路径，值为清理后的文本内容
        """
        processed_texts = {}
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
        """从DOCX文件提取文本
        
        Args:
            file_path: DOCX文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            doc = DocxDocument(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            logger.error(f"Error extracting DOCX content from {file_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """从PDF文件提取文本，使用PyMuPDF (fitz)
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                logger.info(f"Successfully extracted text from PDF using PyMuPDF: {file_path}")
                return text
            else:
                logger.warning(f"No text extracted from PDF: {file_path}")
                return f"[No text extracted from PDF: {os.path.basename(file_path)}]"
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            return f"[PDF提取失败: {os.path.basename(file_path)}]"
    
    def extract_text_from_excel(self, file_path: str) -> str:
        """从Excel文件提取文本
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            wb = load_workbook(file_path, read_only=True, data_only=True)
            text = []
            
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                sheet_text = [f"--- Sheet: {sheet} ---"]
                
                # 读取单元格数据
                for row in ws.rows:
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value))
                    if row_values:
                        sheet_text.append("\t".join(row_values))
                
                if len(sheet_text) > 1:  # 如果有内容
                    text.extend(sheet_text)
                    text.append("")  # 添加空行分隔不同工作表
            
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error extracting Excel content from {file_path}: {e}")
            return ""

if __name__ == "__main__":
    # 测试
    processor = ChineseDocumentProcessor("./test_docs")
    results = processor.process_documents()
    print(f"Processed {len(results)} documents")
    for path, content in results.items():
        print(f"{path}: {len(content)} characters")
