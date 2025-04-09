import os
import re
import torch
import json
import PyPDF2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

class ATSScorer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.stop_words = set(stopwords.words("english"))

    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)

    def get_embedding(self, text):
        if not text:
            return torch.zeros(384)
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

    def extract_keywords(self, text, n=30):
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            word_scores = dict(zip(feature_names, scores))
            top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [word for word, _ in top_keywords]
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            return []

    def find_missing_keywords(self, jd_keywords, resume_text):
        resume_text = resume_text.lower()
        return [kw for kw in jd_keywords if kw.lower() not in resume_text]

    def calculate_similarity_score(self, jd_embedding, resume_embedding):
        jd_embedding = jd_embedding.reshape(1, -1)
        resume_embedding = resume_embedding.reshape(1, -1)
        similarity = cosine_similarity(jd_embedding, resume_embedding)[0][0]
        return round(similarity * 100, 2)

    def analyze_resume(self, resume_text, jd_text):
        if not resume_text or not jd_text:
            return {
                "ats_score": 0,
                "missing_keywords": [],
                "error": "Missing resume or job description text"
            }

        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(jd_text)
        resume_embedding = self.get_embedding(processed_resume)
        jd_embedding = self.get_embedding(processed_jd)
        jd_keywords = self.extract_keywords(jd_text)
        missing_keywords = self.find_missing_keywords(jd_keywords, resume_text)
        ats_score = self.calculate_similarity_score(jd_embedding, resume_embedding)

        return {
            "ats_score": ats_score,
            "missing_keywords": missing_keywords
        }
# ATS class (from ATS_Feat02)
