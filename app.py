from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from .analyzer import ResumeAnalyzer
from .utils import extract_text_from_pdf, extract_text_from_docx
from .models import JobDescription
import os
from dotenv import load_dotenv
import together

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://your-frontend-domain.com"
        ]
    }
})

# Configure Together AI
together.api_key = os.getenv('TOGETHER_API_KEY')

if not together.api_key:
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
        
        # Add error handling for JSON parsing
        try:
            job_desc_dict = json.loads(job_desc_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid job description format'}), 400

        # Extract text based on file type
        if resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file.read())
        elif resume_file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_file.read())
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Parse job description with error handling
        try:
            job_desc = JobDescription(**job_desc_dict)
        except ValueError as e:
            return jsonify({'error': f'Invalid job description data: {str(e)}'}), 400
        
        # Analyze resume
        analysis = analyzer.analyze_resume(resume_text, job_desc)
        
        return jsonify(analysis.dict())
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 