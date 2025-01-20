from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from analyzer import ResumeAnalyzer
from utils import extract_text_from_pdf, extract_text_from_docx
from models import JobDescription
import os
from dotenv import load_dotenv
import together
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://resume-analyzer-front.vercel.app",
            "https://resumebuilder.curiousdevtx.com",
            "http://resumebuilder.curiousdevtx.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Together AI from environment variable
together.api_key = 'ee472495ec6514e25553987a73ed1f43fdfdd77938b4fed33aba5ebc5d7c45bc'
if not together.api_key:
    logger.error("TOGETHER_API_KEY not found in environment variables")
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

analyzer = ResumeAnalyzer()

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        if 'jobDescription' not in request.form:
            return jsonify({'error': 'No job description provided'}), 400

        resume_file = request.files['resume']
        job_desc_data = request.form['jobDescription']
        
        # Log incoming request
        logger.info(f"Processing resume: {resume_file.filename}")
        
        try:
            job_desc_dict = json.loads(job_desc_data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return jsonify({'error': 'Invalid job description format'}), 400

        # Extract text based on file type
        try:
            if resume_file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume_file.read())
            elif resume_file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(resume_file.read())
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            return jsonify({'error': 'Failed to process resume file'}), 500

        # Parse job description
        try:
            job_desc = JobDescription(**job_desc_dict)
        except ValueError as e:
            logger.error(f"Job description validation error: {str(e)}")
            return jsonify({'error': f'Invalid job description data: {str(e)}'}), 400
        
        # Analyze resume
        analysis = analyzer.analyze_resume(resume_text, job_desc)
        logger.info(f"Analysis completed for {resume_file.filename}")
        
        return jsonify(analysis.dict())
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# Don't include this in production
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port) 