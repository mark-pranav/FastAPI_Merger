from fastapi import APIRouter, Form, HTTPException, Depends, status
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from app.services.gemini import process_resumes
import os

router = APIRouter()


class ResumeInput(BaseModel):
    url: HttpUrl
    filename: Optional[str] = Field(None, description="Optional custom filename")
    user_id: str


class ATSRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]


class UserResult(BaseModel):
    user_id: str
    score: float
    content_score: float
    keyword_score: float
    missing_keywords: List[str]
    sgAnalysis: str


class ATSResponse(BaseModel):
    results: List[UserResult]


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured"
        )
    return api_key


@router.post("/analyze", response_model=ATSResponse)
async def analyze_resumes(request: ATSRequest, api_key: str = Depends(get_api_key)):
    if not request.resumes:
        raise HTTPException(status_code=400, detail="No resume URLs provided")

    results = await process_resumes(request.resumes, request.job_description, api_key)
    return {"results": results}
# Gemini ATS routes
