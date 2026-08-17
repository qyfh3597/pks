import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Data preprocessing module for the Personal Knowledge Summary System.
Handles text cleaning, segmentation, and JSONL output.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from html.parser import HTMLParser
import jieba
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from docx import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config
from .logger import get_logger

logger = get_logger()


class HTMLStripper(HTMLParser):
    """Helper class to strip HTML tags."""
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []
    
    def handle_data(self, d):
        self.text.append(d)
    
    def get_data(self):
        return ''.join(self.text)


def strip_html(html: str) -> str:
    """Remove HTML tags from text."""
    stripper = HTMLStripper()
    try:
        stripper.feed(html)
        return stripper.get_data()
    except Exception as e:
        logger.warning(f"Error stripping HTML: {e}")
        return html


def clean_text(text: str, remove_special_chars: bool = True) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Input text
        remove_special_chars: Whether to remove special characters
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_special_chars:
        # Keep only alphanumeric, Chinese characters, and basic punctuation
        text = re.sub(r'[^\w\s\u4e00-\u9fff\.\,\!\?\;\:\-\(\）]', '', text)
    
    return text


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English.
    
    Args:
        text: Input text
        
    Returns:
        'zh' for Chinese, 'en' for English, 'mixed' for mixed
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if chinese_chars > english_chars * 2:
        return 'zh'
    elif english_chars > chinese_chars * 2:
        return 'en'
    else:
        return 'mixed'


def segment_sentences_zh(text: str) -> List[str]:
    """
    Segment Chinese text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split by Chinese punctuation and English periods
    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def segment_sentences_en(text: str) -> List[str]:
    """
    Segment English text into sentences using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(f"Error segmenting sentences: {e}")
        return [text]


def segment_sentences(text: str, language: str = 'auto') -> List[str]:
    """
    Segment text into sentences based on language.
    
    Args:
        text: Input text
        language: 'zh', 'en', or 'auto'
        
    Returns:
        List of sentences
    """
    if language == 'auto':
        language = detect_language(text)
    
    if language == 'zh':
        return segment_sentences_zh(text)
    else:
        return segment_sentences_en(text)


def extract_keywords_zh(text: str, top_k: int = 5) -> List[str]:
    """
    Extract keywords from Chinese text using jieba.
    
    Args:
        text: Input text
        top_k: Number of top keywords to extract
        
    Returns:
        List of keywords
    """
    try:
        words = jieba.cut(text)
        # Filter out single characters and common stopwords
        stopwords = {'的', '了', '是', '在', '和', '有', '一', '这', '不', '人', '中', '大', '为', '上', '个', '国', '我', '以', '要', '他', '时', '来', '用', '们', '生', '到', '作', '地', '于', '出', '就', '分', '对', '成', '会', '可', '主', '发', '年', '动', '同', '工', '也', '能', '下', '过', '民', '前', '面', '方', '法', '其', '长', '多', '经', '学', '制', '进', '或', '行', '等', '种', '高', '该', '她', '而', '开', '所', '还', '产', '后', '家', '尤', '去', '部', '又', '样', '每', '则', '进', '展', '把', '那', '你', '东', '西', '南', '北'}
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords[:top_k]
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return []


def extract_keywords_en(text: str, top_k: int = 5) -> List[str]:
    """
    Extract keywords from English text using simple frequency analysis.
    
    Args:
        text: Input text
        top_k: Number of top keywords to extract
        
    Returns:
        List of keywords
    """
    try:
        words = nltk.word_tokenize(text.lower())
        # Filter out stopwords and short words
        stopwords = set(nltk.corpus.stopwords.words('english'))
        keywords = [w for w in words if len(w) > 3 and w not in stopwords and w.isalpha()]
        
        # Get frequency
        from collections import Counter
        freq = Counter(keywords)
        return [w for w, _ in freq.most_common(top_k)]
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return []


def extract_keywords(text: str, language: str = 'auto', top_k: int = 5) -> List[str]:
    """
    Extract keywords from text based on language.
    
    Args:
        text: Input text
        language: 'zh', 'en', or 'auto'
        top_k: Number of top keywords
        
    Returns:
        List of keywords
    """
    if language == 'auto':
        language = detect_language(text)
    
    if language == 'zh':
        return extract_keywords_zh(text, top_k)
    else:
        return extract_keywords_en(text, top_k)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        document = Document(file_path)
        text = "\n\n".join([paragraph.text for paragraph in document.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return ""

def process_file(
    file_path: str,
    doc_id: str,
    min_text_length: int = 20,
    language: str = 'auto'
) -> List[Dict[str, Any]]:
    """
    Process a single file and extract paragraphs.
    
    Args:
        file_path: Path to the file
        doc_id: Document ID
        min_text_length: Minimum text length for a paragraph
        language: Language for processing
        
    Returns:
        List of processed paragraphs
    """
    records = []
    
    try:
        # Extract title from filename
        title = Path(file_path).stem
        
        # Read file content based on extension
        if file_path.endswith(('.pdf', '.PDF')):
            content = extract_text_from_pdf(file_path)
        elif file_path.endswith(('.docx', '.DOCX')):
            content = extract_text_from_docx(file_path)
        elif file_path.endswith(('.html', '.HTML')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = strip_html(content)
        else: # Default to text-based files (.txt, .md, .markdown)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if not content:
            logger.warning(f"File {file_path} is empty or extraction failed.")
            return records
        
        # Clean text
        content = clean_text(content)
        
        # Detect language
        if language == 'auto':
            language = detect_language(content)
        
        # Split into paragraphs (by double newline or other markers)
        paragraphs = re.split(r'\n\n+', content)
        
        # Process each paragraph
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            # Skip short paragraphs
            if len(paragraph) < min_text_length:
                continue
            
            # Extract keywords
            keywords = extract_keywords(paragraph, language, top_k=5)
            
            # Create record
            record = {
                "doc_id": doc_id,
                "title": title,
                "paragraph_index": para_idx,
                "text": paragraph,
                "keywords": keywords,
                "language": language,
                "length": len(paragraph)
            }
            
            records.append(record)
        
        logger.info(f"Processed {file_path}: {len(records)} paragraphs")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    
    return records


def preprocess_directory(
    input_dir: str,
    output_file: str,
    min_text_length: int = 20,
    language: str = 'auto'
) -> int:
    """
    Process all files in a directory and save to JSONL.
    
    Args:
        input_dir: Input directory containing text files
        output_file: Output JSONL file path
        min_text_length: Minimum text length for a paragraph
        language: Language for processing
        
    Returns:
        Total number of records processed
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 0
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    doc_id_counter = 0
    
    # Process all text files
    file_extensions = ['.txt', '.md', '.html', '.markdown', '.pdf', '.docx']
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in sorted(input_path.rglob('*')):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                doc_id = f"doc_{doc_id_counter:04d}"
                records = process_file(
                    str(file_path),
                    doc_id,
                    min_text_length,
                    language
                )
                
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_records += 1
                
                doc_id_counter += 1
    
    logger.info(f"Preprocessing complete: {total_records} total records saved to {output_file}")
    return total_records


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Preprocess text data for the Personal Knowledge Summary System"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw/",
        help="Input directory containing raw text files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/processed.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum text length for a paragraph"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=['zh', 'en', 'auto'],
        help="Language for processing"
    )
    
    args = parser.parse_args()
    
    total = preprocess_directory(
        args.input,
        args.output,
        args.min_length,
        args.language
    )
    
    print(f"\nPreprocessing complete: {total} records saved")


if __name__ == "__main__":
    main()
