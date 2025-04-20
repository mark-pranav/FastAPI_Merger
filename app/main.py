from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import recommender, ats_scorer, gemini_ats
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Unified ATS and Recommendation API",
    description="Combines job recommendation, ATS scoring, and Gemini-based resume analysis.",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(recommender.router, prefix="/api/feature", tags=["Recommendation"])
app.include_router(ats_scorer.router, prefix="/api/feature", tags=["ATS Scoring"])
app.include_router(gemini_ats.router, prefix="/api/feature", tags=["Gemini Analysis"])

@app.get("/health")
def health():
    return {"status": "running"}
# Entry point for FastAPI app
