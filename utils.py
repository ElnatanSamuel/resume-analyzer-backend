from pdfminer.high_level import extract_text
from docx import Document
import io
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from pdfminer.layout import LAParams
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text: str) -> np.ndarray:
    """Get embeddings for text using sentence transformer"""
    return model.encode([text])[0]

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file with improved layout handling
    """
    try:
        text = extract_text(
            io.BytesIO(pdf_bytes),
            laparams=LAParams(
                char_margin=1.0,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True
            )
        )
        if not text.strip():
            raise ValueError("Extracted text is empty")
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise ValueError(f"Error processing PDF: {str(e)}")

def extract_text_from_docx(docx_bytes: bytes) -> str:
    """
    Extract text from a DOCX file
    """
    try:
        doc = Document(io.BytesIO(docx_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            raise ValueError("Extracted text is empty")
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise ValueError(f"Error processing DOCX: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text with improved handling
    """
    # Remove extra whitespace and normalize spaces
    text = " ".join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace("\x0c", " ")  # Form feed
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Remove multiple spaces and normalize newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any remaining special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;!?-]', '', text)
    
    return text.strip()

def extract_sections(text: str) -> dict:
    """Extract different sections from resume text with improved detection"""
    sections = {
        'skills': [],
        'experience': [],
        'education': [],
        'projects': []
    }
    
    # Enhanced section keywords with more variations
    section_keywords = {
        'skills': ['skills', 'technical skills', 'technologies', 'tech stack', 'competencies', 
                  'expertise', 'proficiencies', 'qualifications'],
        'experience': ['experience', 'work experience', 'employment', 'work history', 
                      'professional experience', 'career history'],
        'education': ['education', 'academic', 'qualifications', 'academic background',
                     'educational background'],
        'projects': ['projects', 'personal projects', 'key projects', 'professional projects',
                    'portfolio']
    }
    
    # Split text into lines and normalize
    lines = [line.strip().lower() for line in text.split('\n') if line.strip()]
    current_section = None
    current_content = []
    
    # First pass: identify section boundaries
    section_boundaries = []
    for i, line in enumerate(lines):
        for section, keywords in section_keywords.items():
            if any(k in line for k in keywords):
                section_boundaries.append((i, section))
    
    # Sort boundaries by line number
    section_boundaries.sort()
    
    # Second pass: extract content between boundaries
    for i in range(len(section_boundaries)):
        start_idx = section_boundaries[i][0]
        section = section_boundaries[i][1]
        
        # Get end index (either next section or end of text)
        end_idx = section_boundaries[i + 1][0] if i < len(section_boundaries) - 1 else len(lines)
        
        # Extract content for this section
        content = lines[start_idx + 1:end_idx]  # Skip the section header
        sections[section].extend(content)
    
    # Debug output
    print("Found sections:", {k: len(v) for k, v in sections.items()})
    for section, content in sections.items():
        print(f"\n{section.upper()} section first 100 chars:", ' '.join(content)[:100])
    
    return sections

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    emb1 = get_embeddings(text1)
    emb2 = get_embeddings(text2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)) 