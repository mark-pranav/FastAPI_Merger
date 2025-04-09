from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from PIL import Image
import io
import asyncio
import pytesseract

from app.services.ats import ATSScorer

router = APIRouter()
ats_scorer = ATSScorer()

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Update this for deployment

class ResumeInput(BaseModel):
    user_id: str
    resume_url: str

class ATSRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]

class ATSResult(BaseModel):
    user_id: str
    ats_score: float
    missing_keywords: List[str]
    error: Optional[str] = None


async def fetch_resume_text_from_img(url: str, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise ValueError(f"OCR failed or invalid URL: {e}")


async def process_resume(resume: ResumeInput, jd_text: str):
    try:
        resume_text = await fetch_resume_text_from_img(resume.resume_url)
        result = ats_scorer.analyze_resume(resume_text, jd_text)
        return {
            "user_id": resume.user_id,
            "ats_score": result["ats_score"],
            "missing_keywords": result["missing_keywords"]
        }
    except Exception as e:
        return {
            "user_id": resume.user_id,
            "ats_score": 0,
            "missing_keywords": [],
            "error": str(e)
        }


@router.post("/score", response_model=List[ATSResult])
async def score_resumes(data: ATSRequest):
    try:
        tasks = [process_resume(resume, data.job_description) for resume in data.resumes]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ATS Scorer routes
