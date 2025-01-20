from pydantic import BaseModel, Field
from typing import List, Optional

class JobDescription(BaseModel):
    text: str = Field(..., min_length=50, description="The full text of the job description")
    required_skills: List[str] = Field(..., min_items=1, description="List of required skills for the position")
    preferred_skills: List[str] = Field(default=[], description="List of preferred/optional skills for the position")

    class Config:
        schema_extra = {
            "example": {
                "text": "We are looking for a Python developer with experience in web development...",
                "required_skills": ["python", "flask", "sql"],
                "preferred_skills": ["docker", "aws", "react"]
            }
        }

class ResumeAnalysis(BaseModel):
    match_percentage: float = Field(..., ge=0, le=100, description="Percentage match between resume and job requirements")
    skill_score: float = Field(..., ge=0, le=100, description="Score based on skill matches")
    experience_relevance: float = Field(..., ge=0, le=100, description="Relevance of experience to the job")
    overall_score: float = Field(..., ge=0, le=100, description="Overall resume score")
    missing_skills: List[str] = Field(..., description="Skills required but not found")
    matching_skills: List[str] = Field(..., description="Matching skills found")
    score_explanation: str = Field(..., description="Explanation of the score calculation")
    suggestions: List[dict] = Field(..., description="Formatted suggestions for improvement")

    class Config:
        schema_extra = {
            "example": {
                "match_percentage": 75.5,
                "skill_score": 80.0,
                "experience_relevance": 70.0,
                "overall_score": 75.2,
                "missing_skills": ["docker"],
                "matching_skills": ["python", "flask"],
                "score_explanation": "The score is calculated based on the percentage match, skill matches, experience relevance, and overall resume score.",
                "suggestions": [
                    {
                        "type": "project",
                        "title": "Suggested Projects",
                        "items": ["Build a containerized web application using Docker"]
                    }
                ]
            }
        } 