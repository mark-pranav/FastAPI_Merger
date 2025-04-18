from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import json
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

router = APIRouter()

# Download required NLTK data (ensure it is done only once in actual deployment)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
nltk.data.path.append('./nltk_data')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


nlp = spacy.load("en_core_web_sm")

# Data Models
class Student(BaseModel):
    name: str
    skills: List[str]

class Internship(BaseModel):
    title: str
    company: str
    location: str
    requirements: List[str]
    jd: str

class RecommendationRequest(BaseModel):
    student: Student
    internships: List[Internship]
    top_n: Optional[int] = 3
    min_similarity: Optional[float] = 0.1

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    matching_scores: Dict[str, float]

# Helpers
def extract_keywords_from_jd(jd: str) -> List[str]:
    doc = nlp(jd.lower())
    keywords = []

    for chunk in doc.noun_chunks:
        if chunk.text.strip() not in stopwords.words("english"):
            keywords.append(chunk.text.strip())

    relevant_entities = ["SKILL", "ORG", "PRODUCT", "GPE", "LANGUAGE"]
    for ent in doc.ents:
        if ent.label_ in relevant_entities:
            keywords.append(ent.text.strip())

    for token in doc:
        if token.pos_ == "VERB" and token.text not in stopwords.words("english"):
            keywords.append(token.text)

    return list(set(keywords))

def calculate_skill_match_score(student_skills, internship_reqs, jd_keywords) -> Dict:
    student_skills_set = set(student_skills)
    requirements_set = set(internship_reqs)
    jd_keywords_set = set(jd_keywords)

    direct_matches = student_skills_set & requirements_set
    keyword_matches = student_skills_set & jd_keywords_set

    return {
        "direct_match_score": len(direct_matches) / len(requirements_set) if requirements_set else 0,
        "keyword_match_score": len(keyword_matches) / len(jd_keywords_set) if jd_keywords_set else 0,
        "direct_matches": list(direct_matches),
        "keyword_matches": list(keyword_matches),
    }

# Endpoints
@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        internships_data = []
        for internship in request.internships:
            jd_keywords = extract_keywords_from_jd(internship.jd)
            internships_data.append({
                "title": internship.title,
                "company": internship.company,
                "location": internship.location,
                "requirements": " ".join(internship.requirements),
                "jd_keywords": jd_keywords,
                "combined_text": f"{' '.join(internship.requirements)} {' '.join(jd_keywords)}"
            })

        internships_df = pd.DataFrame(internships_data)

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        internship_vectors = vectorizer.fit_transform(internships_df["combined_text"])
        student_vector = vectorizer.transform([" ".join(request.student.skills)])

        similarity_scores = np.asarray(student_vector.dot(internship_vectors.T).todense())[0]
        top_indices = (-similarity_scores).argsort()[:request.top_n]

        recommendations = []
        for idx in top_indices:
            if similarity_scores[idx] >= request.min_similarity:
                internship = internships_df.iloc[idx]
                skill_matches = calculate_skill_match_score(
                    request.student.skills,
                    internship["requirements"].split(),
                    internship["jd_keywords"]
                )

                final_score = (
                    similarity_scores[idx] * 0.4 +
                    skill_matches["direct_match_score"] * 0.4 +
                    skill_matches["keyword_match_score"] * 0.2
                )

                recommendations.append({
                    "title": internship["title"],
                    "company": internship["company"],
                    "location": internship["location"],
                    "similarity_score": float(final_score),
                    "tfidf_similarity": float(similarity_scores[idx]),
                    "direct_match_score": skill_matches["direct_match_score"],
                    "keyword_match_score": skill_matches["keyword_match_score"],
                    "direct_matches": skill_matches["direct_matches"],
                    "keyword_matches": skill_matches["keyword_matches"]
                })

        return {
            "recommendations": recommendations,
            "matching_scores": {
                "overall_match": float(np.mean([r['similarity_score'] for r in recommendations])) if recommendations else 0,
                "skill_match": float(np.mean([r['direct_match_score'] for r in recommendations])) if recommendations else 0,
                "keyword_match": float(np.mean([r['keyword_match_score'] for r in recommendations])) if recommendations else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-internships/")
async def upload_internships(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        internships = json.loads(contents)
        return {"message": f"Successfully loaded {len(internships)} internships"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# Job Recommendation routes
