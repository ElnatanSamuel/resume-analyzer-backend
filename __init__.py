"""
AI-Powered Resume Analyzer Backend
A Flask application for analyzing resumes against job descriptions using NLP.
"""

__version__ = "1.0.0"

from .models import JobDescription, ResumeAnalysis
from .analyzer import ResumeAnalyzer
from .utils import extract_text_from_pdf, extract_text_from_docx, clean_text

__all__ = [
    'JobDescription',
    'ResumeAnalysis',
    'ResumeAnalyzer',
    'extract_text_from_pdf',
    'extract_text_from_docx',
    'clean_text',
] 