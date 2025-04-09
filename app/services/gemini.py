import os
import json
import re
import asyncio
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List

load_dotenv()
logger = logging.getLogger("gemini")

SKILL_VARIATIONS = {
    # JavaScript frameworks/libraries
    "react": ["reactjs", "react.js", "react js"],
    "angular": ["angularjs", "angular.js", "angular js"],
    "vue": ["vuejs", "vue.js", "vue js"],
    "node": ["nodejs", "node.js", "node js"],

    # Programming languages
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "python": ["py"],
    "java": ["jdk"],
    "c#": ["csharp", "c sharp"],
    "c++": ["cpp", "cplusplus", "c plus plus"],

    # Database technologies
    "postgresql": ["postgres", "pgsql"],
    "mongodb": ["mongo"],
    "mysql": ["sql"],
    "mssql": ["sql server", "microsoft sql server"],

    # Cloud platforms
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform", "google cloud"],
    "azure": ["microsoft azure"],

    # Big data technologies
    "hadoop": ["apache hadoop"],
    "spark": ["apache spark"],
    "kafka": ["apache kafka"],

    # DevOps tools
    "docker": ["containerization"],
    "kubernetes": ["k8s"],
    "jenkins": ["ci/cd", "cicd"],
    "terraform": ["infrastructure as code", "iac"],

    # Mobile development
    "react native": ["reactnative"],
    "flutter": ["dart flutter"],
    "swift": ["ios development"],
    "kotlin": ["android development"],

    # AI/ML
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],

    # Others
    "restful api": ["rest api", "rest", "restful"],
    "graphql": ["gql"],
    "html5": ["html"],
    "css3": ["css"],
    "sass": ["scss"],
    "less": ["css preprocessor"],
}

REVERSE_SKILL_MAP = {}
for key, variations in SKILL_VARIATIONS.items():
    REVERSE_SKILL_MAP[key] = key
    for var in variations:
        REVERSE_SKILL_MAP[var] = key

def normalize_keyword(keyword):
    keyword = keyword.strip().lower()
    return REVERSE_SKILL_MAP.get(keyword, keyword)

def prepare_image_prompt(job_description):
    return f"""
    You are looking at a resume image. First, extract all the text content from the resume.

    Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in Technical fields like:
    
    CRITICALLY IMPORTANT: Each resume must be evaluated individually and given a unique score based on its specific content.
    
    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering
    - Frontend Developer
    - Backend Developer
    - DevOps Engineer
    - Programming Specialist

    Evaluate the resume against the job description using this scoring system:
    - ContentMatch (0-50 points): Provide a precise, differentiated score for how well this specific candidate's experience aligns with job requirements
    - KeywordMatch (0-50 points): Count the actual number of relevant keywords present and score accordingly
    
    Be extremely discriminating in your scoring. Even similar resumes should receive different scores based on subtle differences in experience, relevance, and keyword matches.
    
    IMPORTANT: When matching keywords, skills, and technologies, be intelligent about variations:
    - Consider "React", "ReactJS", and "React.js" as the same technology
    - Recognize when technologies are mentioned with slight variations (like "Node.js" vs "Node")
    - Match skill abbreviations with their full names (like "ML" with "Machine Learning")
    - Don't penalize for these variations - they should count as matches, not missing keywords

    Consider that the job market is highly competitive. Provide detailed feedback for resume improvement.

    Job Description:
    {job_description}

    Provide a response in the following JSON format ONLY, with no additional text:
    {{
        "ContentMatch": "Score (0-50) for overall content/experience alignment with job description",
        "KeywordMatch": "Score (0-50) for matching of specific keywords and skills",
        "TotalScore": "Sum of ContentMatch and KeywordMatch (0-100)",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "A concise 3-sentence evaluation highlighting strengths, key gaps, and actionable improvement suggestions.",
        "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
    }}

    IMPORTANT: The field "Skills Gap Analysis" MUST be exactly 3 sentences long, no more and no less, and must use exactly that field name with spaces.
    """

def extract_score_value(raw):
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"(\d+\.?\d*)", str(raw))
    return float(match.group(1)) if match else 0.0

def process_missing_keywords(keywords):
    if isinstance(keywords, str):
        try:
            keywords = json.loads(keywords)
        except:
            keywords = [kw.strip() for kw in keywords.split(",")]
    return list({normalize_keyword(k) for k in keywords})

def extract_scores(response):
    content = extract_score_value(response.get("ContentMatch", 0))
    keyword = extract_score_value(response.get("KeywordMatch", 0))
    total = extract_score_value(response.get("TotalScore", 0))
    total = content + keyword if abs(total - (content + keyword)) > 5 else total
    return dict(content_score=content, keyword_score=keyword, total_score=total)

def extract_json_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group()) if match else {}

async def get_gemini_response(prompt, image_url, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, image_url])
    return extract_json_response(response.text)

async def process_resume_url(resume, job_description, api_key):
    try:
        url = str(resume.url)
        user_id = resume.user_id
        prompt = prepare_image_prompt(job_description)
        response = await get_gemini_response(prompt, url, api_key)
        scores = extract_scores(response)
        keywords = process_missing_keywords(response.get("MissingKeywords", []))
        sg_analysis = next((response[k] for k in response if k.lower().replace(" ", "") == "skillsgapanalysis"), "")
        return {
            "user_id": user_id,
            "score": scores["total_score"],
            "content_score": scores["content_score"],
            "keyword_score": scores["keyword_score"],
            "missing_keywords": keywords,
            "sgAnalysis": sg_analysis or "Not available"
        }
    except Exception as e:
        return {
            "user_id": resume.user_id,
            "score": 0.0,
            "content_score": 0.0,
            "keyword_score": 0.0,
            "missing_keywords": [],
            "sgAnalysis": "Error: " + str(e)
        }

async def process_resumes(resumes: List, job_description: str, api_key: str):
    sem = asyncio.Semaphore(3)
    async def sem_wrap(resume):
        async with sem:
            return await process_resume_url(resume, job_description, api_key)
    tasks = [sem_wrap(r) for r in resumes]
    return await asyncio.gather(*tasks)
# Gemini Vision logic
