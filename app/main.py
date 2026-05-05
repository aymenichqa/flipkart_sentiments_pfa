# app/main.py — API FastAPI complète
# Lancer avec : uvicorn app.main:app --reload --port 8000

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib, sys, os, time, requests
from bs4 import BeautifulSoup
from datetime import datetime

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)
from src.preprocessing import clean_text

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flipkart Sentiment Analysis API",
    description="""
    API de classification de sentiment pour reviews e-commerce.
    Supporte : Trustpilot, Amazon.in, textes libres.
    Modèles : Logistic Regression, Naive Bayes, LinearSVC.
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement des modèles (une seule fois au démarrage) ─────────────────────
vectorizer = None
modele     = None

@app.on_event("startup")
def load_models():
    global vectorizer, modele
    models_path = os.path.join(BASE_PATH, 'models')
    vectorizer = joblib.load(os.path.join(models_path, 'tfidf_vectorizer.pkl'))
    modele     = joblib.load(os.path.join(models_path, 'best_model_champion.pkl'))
    print(f"✅ Modèle chargé : {type(modele).__name__}")
    print(f"✅ Vectorizer    : {vectorizer.get_params()['max_features']} features TF-IDF")

# ── Schémas de données ────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class ScrapeRequest(BaseModel):
    source: str          # "trustpilot" ou "amazon"
    target: str          # nom de l'entreprise ou ASIN
    max_pages: int = 3   # limiter pour la démo

class ReviewResult(BaseModel):
    text_original: str
    sentiment: str
    confidence: float
    proba_positif: float
    proba_neutre: float
    proba_negatif: float

# ── Fonction de prédiction centrale ──────────────────────────────────────────
def predict_one(text: str) -> dict:
    """Pipeline complet : texte brut → sentiment + probabilités."""
    cleaned  = clean_text(text)
    X        = vectorizer.transform([cleaned])
    label    = modele.predict(X)[0]
    probas   = modele.predict_proba(X)[0]
    classes  = list(modele.classes_)
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, probas)}
    return {
        "sentiment"    : label,
        "confidence"   : round(float(max(probas)), 4),
        "probabilities": proba_dict,
        "proba_POSITIF": proba_dict.get("POSITIF", 0),
        "proba_NEUTRE" : proba_dict.get("NEUTRE", 0),
        "proba_NEGATIF": proba_dict.get("NEGATIF", 0),
    }

# ── Scrapers intégrés ─────────────────────────────────────────────────────────
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def scrape_trustpilot(company: str, max_pages: int = 3) -> List[dict]:
    """Scrape les reviews Trustpilot d'une entreprise."""
    from app.scraper import scrape_trustpilot
    return scrape_trustpilot(company, max_pages)


def scrape_amazon(asin: str, max_pages: int = 2) -> List[dict]:
    """Scrape les reviews Amazon.in d'un produit par ASIN."""
    from app.scraper import scrape_amazon
    return scrape_amazon(asin, max_pages)

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message" : "Flipkart Sentiment Analysis API v2.0",
        "endpoints": ["/health", "/predict", "/predict/batch", "/scrape", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status"     : "ok",
        "model"      : type(modele).__name__,
        "tfidf_features": vectorizer.get_params()['max_features'],
        "timestamp"  : datetime.now().isoformat()
    }

# --- Prédiction texte unique ---
@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
    result = predict_one(req.text)
    result["text_original"] = req.text[:200]
    return result

# --- Prédiction batch ---
@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if len(req.texts) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 textes par batch")
    results = []
    for text in req.texts:
        pred = predict_one(text)
        pred["text_original"] = text[:100]
        results.append(pred)

    # Statistiques globales
    sentiments = [r["sentiment"] for r in results]
    stats = {
        "total"   : len(results),
        "POSITIF" : sentiments.count("POSITIF"),
        "NEUTRE"  : sentiments.count("NEUTRE"),
        "NEGATIF" : sentiments.count("NEGATIF"),
        "confidence_moyenne": round(sum(r["confidence"] for r in results) / len(results), 4)
    }
    return {"results": results, "statistics": stats}

# --- Endpoint principal : scraper + analyser ---
@app.post("/scrape")
def scrape_and_analyze(req: ScrapeRequest):
    """
    Scrape les reviews d'un site et retourne l'analyse de sentiment.
    source = "trustpilot" | "amazon"
    target = nom d'entreprise (trustpilot) ou ASIN (amazon)
    """
    # 1. Scraper
    if req.source == "trustpilot":
        raw_reviews = scrape_trustpilot(req.target, max_pages=req.max_pages)
    elif req.source == "amazon":
        raw_reviews = scrape_amazon(req.target, max_pages=req.max_pages)
    else:
        raise HTTPException(status_code=400,
                            detail="source doit être 'trustpilot' ou 'amazon'")

    if not raw_reviews:
        raise HTTPException(status_code=404,
                            detail=f"Aucune review trouvée pour '{req.target}'. "
                                   "Vérifier le nom d'entreprise ou l'ASIN.")

    # 2. Analyser chaque review
    analyzed = []
    for review in raw_reviews:
        pred = predict_one(review["text"])
        analyzed.append({
            **review,
            "sentiment"    : pred["sentiment"],
            "confidence"   : pred["confidence"],
            "proba_POSITIF": pred["proba_POSITIF"],
            "proba_NEUTRE" : pred["proba_NEUTRE"],
            "proba_NEGATIF": pred["proba_NEGATIF"],
        })

    # 3. Statistiques globales
    sentiments = [r["sentiment"] for r in analyzed]
    n = len(analyzed)
    stats = {
        "total"              : n,
        "source"             : req.source,
        "target"             : req.target,
        "POSITIF"            : sentiments.count("POSITIF"),
        "NEUTRE"             : sentiments.count("NEUTRE"),
        "NEGATIF"            : sentiments.count("NEGATIF"),
        "pct_positif"        : round(sentiments.count("POSITIF") / n * 100, 1),
        "pct_negatif"        : round(sentiments.count("NEGATIF") / n * 100, 1),
        "confidence_moyenne" : round(sum(r["confidence"] for r in analyzed) / n, 4),
        "alerte"             : sentiments.count("NEGATIF") / n > 0.3,
        "timestamp"          : datetime.now().isoformat(),
    }

    return {"statistics": stats, "reviews": analyzed}