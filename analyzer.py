from transformers import pipeline
from typing import List, Dict
import torch
from .models import JobDescription, ResumeAnalysis
import logging
import re
import os
import together
from backend.utils import extract_sections, get_embeddings
import numpy as np

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        # Configure Together AI
        together.api_key =os.environ.get('TOGETHER_API_KEY')
        self.llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        # Expanded skill relationships for various job types
        self.skill_relationships = {
            # Technical Skills
            "react": ["javascript", "html", "css", "web development", "frontend"],
            "python": ["programming", "coding", "software development", "scripting"],
            "java": ["programming", "coding", "software development", "object-oriented"],
            
            # Business/Management Skills
            "project management": ["leadership", "agile", "scrum", "team management", "planning"],
            "business analysis": ["requirements gathering", "stakeholder management", "documentation"],
            "marketing": ["digital marketing", "social media", "content creation", "analytics"],
            
            # Creative Skills
            "graphic design": ["adobe creative suite", "illustration", "visual design", "typography"],
            "content writing": ["copywriting", "editing", "blogging", "seo"],
            "ui/ux design": ["user research", "wireframing", "prototyping", "usability"],
            
            # Soft Skills
            "leadership": ["team management", "decision making", "mentoring", "strategy"],
            "communication": ["presentation", "writing", "interpersonal", "public speaking"],
            "problem solving": ["analytical thinking", "critical thinking", "troubleshooting"],
            
            # Healthcare Skills
            "patient care": ["medical terminology", "healthcare", "clinical experience"],
            "nursing": ["patient assessment", "medical procedures", "healthcare"],
            
            # Finance Skills
            "financial analysis": ["excel", "modeling", "forecasting", "budgeting"],
            "accounting": ["bookkeeping", "financial reporting", "tax preparation"],
            
            # Sales Skills
            "sales": ["negotiation", "client relationship", "business development"],
            "customer service": ["client support", "problem resolution", "communication"]
        }
        
    def get_related_skills(self, skill: str) -> List[str]:
        """Get related skills for a given skill"""
        skill = skill.lower()
        related = self.skill_relationships.get(skill, [])
        return [skill] + related

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def extract_skills(self, resume_text: str) -> List[str]:
        # Clean and normalize text
        cleaned_text = self.clean_text(resume_text)
        sections = extract_sections(cleaned_text)
        
        detected_skills = set()
        
        # Process all text including section headers
        all_text = cleaned_text.lower()
        
        print("Full text length:", len(all_text))
        print("Sample of text:", all_text[:200])
        
        # Direct keyword matching with fuzzy matching
        for skill, related in self.skill_relationships.items():
            skill_lower = skill.lower()
            
            # Check for skill in the entire text
            if skill_lower in all_text:
                detected_skills.add(skill)
                detected_skills.update(related)
                print(f"Found skill: {skill} (exact match)")
                continue
            
            # Check for skill parts (for compound skills)
            skill_parts = skill_lower.split()
            if len(skill_parts) > 1 and all(part in all_text for part in skill_parts):
                detected_skills.add(skill)
                detected_skills.update(related)
                print(f"Found skill: {skill} (parts match)")
                continue
            
            # Check for related skills
            if any(rel.lower() in all_text for rel in related):
                detected_skills.add(skill)
                detected_skills.update(related)
                print(f"Found skill: {skill} (related match)")
        
        # Add common variations
        tech_variations = {
            'reactjs': 'react',
            'react.js': 'react',
            'nodejs': 'node',
            'node.js': 'node',
            'next.js': 'nextjs',
            'expressjs': 'express',
            'express.js': 'express'
        }
        
        for var, main_skill in tech_variations.items():
            if var in all_text:
                detected_skills.add(main_skill)
                print(f"Found skill: {main_skill} (variation match)")
        
        print("Final detected skills:", detected_skills)
        return list(detected_skills)

    def analyze_resume(self, resume_text: str, job_description: JobDescription) -> ResumeAnalysis:
        try:
            # Extract skills with debugging
            print("Processing resume text...")
            resume_skills = self.extract_skills(resume_text)
            resume_skills_lower = [s.lower() for s in resume_skills]
            print(f"Found resume skills: {resume_skills}")
            
            # Compare skills with more flexible matching
            matching_skills = set()
            missing_skills = set()
            matched_skill_count = 0
            
            print(f"Required skills: {job_description.required_skills}")
            
            for skill in job_description.required_skills:
                skill_lower = skill.lower()
                related_skills = self.get_related_skills(skill)
                related_skills_lower = [s.lower() for s in related_skills]
                
                # Check for skill matches including variations
                if any(rs in resume_skills_lower for rs in related_skills_lower) or \
                   any(rs.replace(' ', '') in ' '.join(resume_skills_lower) for rs in related_skills_lower):
                    matching_skills.update(related_skills)
                    matched_skill_count += 1
                    print(f"Matched skill: {skill}")
                else:
                    missing_skills.update(related_skills)
                    print(f"Missing skill: {skill}")
            
            # Calculate scores
            total_skills = len(job_description.required_skills)
            skill_match = (matched_skill_count / total_skills * 100) if total_skills > 0 else 0
            
            # Generate score explanation
            score_explanation = f"""
Score Breakdown:
• Found {len(matching_skills)} relevant skills
• Missing {len(missing_skills)} required skills
• Matched {matched_skill_count} out of {total_skills} required skill categories
• Raw match score: {skill_match:.1f}%
"""
            
            # Calculate experience relevance
            relevance_score = self.calculate_experience_relevance(resume_text, job_description.text)
            
            # Calculate overall score (weighted average)
            overall_score = (skill_match * 0.6) + (relevance_score * 0.4)
            
            # Get formatted suggestions with improved prompt
            suggestions = self.get_formatted_suggestions(
                resume_text,
                job_description,
                list(missing_skills),
                overall_score
            )
            
            return ResumeAnalysis(
                match_percentage=skill_match,
                skill_score=skill_match,
                experience_relevance=relevance_score,
                overall_score=overall_score,
                missing_skills=list(missing_skills),
                matching_skills=list(matching_skills),
                score_explanation=score_explanation,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in analyze_resume: {str(e)}")
            raise ValueError(f"Failed to analyze resume: {str(e)}")

    def get_together_suggestions(self, resume_text: str, job_description: JobDescription, 
                               missing_skills: List[str], match_percentage: float) -> List[str]:
        """Get personalized suggestions from Together AI"""
        prompt = f"""
As a career advisor, analyze this situation and provide specific suggestions:

Job Requirements: {job_description.text}
Missing Skills: {', '.join(missing_skills)}
Current Match: {match_percentage}%

Please provide:
1. 2-3 specific project ideas to gain the missing skills
2. Suggestions for improving the resume
3. Learning resources or certifications to consider

Format your response as a bullet-point list, keeping suggestions concise and actionable. 
"""
        
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=self.llm_model,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.1,
                stop=['</s>']
            )
            
            # Parse response and clean suggestions
            suggestions = response['output']['choices'][0]['text'].strip().split('\n')
            print(suggestions)
            # Clean and filter suggestions
            suggestions = [s.strip().lstrip('•-* ') for s in suggestions if s.strip() and not s.startswith(('1.', '2.', '3.'))]
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting Together AI suggestions: {str(e)}")
            return ["Consider working on projects to gain experience with: " + ', '.join(missing_skills)]

    def calculate_experience_relevance(self, resume_text: str, job_desc: str) -> float:
        """Calculate how relevant the experience is to the job using text chunks"""
        try:
            # Split texts into smaller chunks
            max_length = 512  # Safe length for BERT-based models
            resume_chunks = [resume_text[i:i+max_length] for i in range(0, len(resume_text), max_length)]
            job_chunks = [job_desc[i:i+max_length] for i in range(0, len(job_desc), max_length)]
            
            # Calculate relevance for each chunk combination
            scores = []
            for r_chunk in resume_chunks:
                for j_chunk in job_chunks:
                    if not r_chunk.strip() or not j_chunk.strip():
                        continue
                    result = self.model(
                        r_chunk,
                        candidate_labels=[j_chunk],
                        hypothesis_template="This experience is relevant to: {}"
                    )
                    scores.append(result['scores'][0])
            
            # Return the maximum relevance score found
            return max(scores) * 100 if scores else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            return 50.0

    def get_formatted_suggestions(self, resume_text: str, job_description: JobDescription, 
                                missing_skills: List[str], current_score: float) -> List[dict]:
        # Extract additional skills from job description text
        desc_text = job_description.text.lower()
        all_skills = set(job_description.required_skills + job_description.preferred_skills)
        
        # Extract context skills using the new method
        context_skills = self.extract_context_skills(desc_text)
        potential_skills = [skill for skill in context_skills if skill not in all_skills]
        
        prompt = f"""
        As a career advisor, analyze this situation and provide targeted suggestions:

        Job Description: {job_description.text}
        Required Skills: {', '.join(job_description.required_skills)}
        Preferred Skills: {', '.join(job_description.preferred_skills)}
        Context Skills Found: {', '.join(potential_skills)}
        Missing Skills: {', '.join(missing_skills)}
        Current Match: {current_score}%

        Provide strictly relevant suggestions in these categories:
        1. Projects (start with 'Project:')
        - Suggest 2-3 projects that specifically target the required skills AND context skills
        - Projects should incorporate: {', '.join(job_description.required_skills)} and {', '.join(potential_skills)}
        - Focus on modern development practices mentioned in the job description
        
        2. Resume Improvements (start with 'Resume:')
        - Suggest how to better highlight experience with ALL skills (required, preferred, and contextual)
        - Include specific formatting tips for highlighting: {', '.join(job_description.required_skills + potential_skills)}
        - Focus on modern tech stack and testing experience if mentioned
        
        3. Learning Path (start with 'Learning:')
        - Recommend specific courses/certifications for:
          * Missing required skills: {', '.join(missing_skills)}
          * Context skills from description: {', '.join(potential_skills)}
        - Include resources for testing frameworks and modern development practices if mentioned
        
        Keep suggestions focused on both explicit requirements AND contextual skills from the job description.
        """
        
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=self.llm_model,
                max_tokens=1500,  # Increased for more detailed responses
                temperature=0.6,
                top_p=0.7,
                top_k=40,
                repetition_penalty=1.2,
                stop=['</s>']
            )
            
            raw_suggestions = response['output']['choices'][0]['text'].strip().split('\n')
            
            # Initialize categories
            formatted_suggestions = {
                "projects": {"type": "projects", "title": "Recommended Projects", "items": []},
                "resume": {"type": "resume", "title": "Resume Improvements", "items": []},
                "learning": {"type": "learning", "title": "Learning Path", "items": []}
            }
            
            current_category = None
            
            # Process suggestions with better categorization
            for line in raw_suggestions:
                line = line.strip().lstrip('•-* ')
                if not line:
                    continue
                    
                if line.lower().startswith('project:'):
                    current_category = "projects"
                    line = line.replace('Project:', '').strip()
                elif line.lower().startswith('resume:'):
                    current_category = "resume"
                    line = line.replace('Resume:', '').strip()
                elif line.lower().startswith('learning:'):
                    current_category = "learning"
                    line = line.replace('Learning:', '').strip()
                
                if current_category and line:
                    formatted_suggestions[current_category]["items"].append(line)
            
            # Return suggestions in consistent order
            return [
                formatted_suggestions["projects"],
                formatted_suggestions["resume"],
                formatted_suggestions["learning"]
            ]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return [
                {
                    "type": "projects",
                    "title": "Recommended Projects",
                    "items": [f"Build projects focusing on: {', '.join(missing_skills)}"]
                },
                {
                    "type": "resume",
                    "title": "Resume Improvements",
                    "items": ["Highlight your experience with the required skills"]
                },
                {
                    "type": "learning",
                    "title": "Learning Path",
                    "items": [f"Focus on learning: {', '.join(missing_skills)}"]
                }
            ]

    def extract_context_skills(self, text: str) -> List[str]:
        """Extract skills mentioned in context but not explicitly listed"""
        # Common tech terms and variations
        tech_patterns = [
            r'\b(?:react\.?js|next\.?js|node\.?js|express\.?js)\b',  # JS ecosystem
            r'\b(?:jest|cypress|selenium|mocha|chai)\b',  # Testing
            r'\b(?:docker|kubernetes|aws|azure|gcp)\b',  # DevOps
            r'\b(?:sql|mongodb|postgres|mysql)\b',  # Databases
            r'\b(?:python|java|golang|rust|cpp)\b',  # Programming languages
            r'\b(?:ci/cd|git|github|gitlab)\b',  # Version control/DevOps
            r'\b(?:html5|css3|sass|less|tailwind)\b',  # Frontend
            r'\b(?:agile|scrum|kanban)\b',  # Methodologies
        ]
        
        found_skills = set()
        text = text.lower()
        
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                found_skills.add(match.group())
        
        return list(found_skills)