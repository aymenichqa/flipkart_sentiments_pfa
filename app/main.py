# app/main.py — API FastAPI v2.1
# Lancer depuis la RACINE du projet :  uvicorn app.main:app --reload --port 8000
#
# Corrections v2.1 :
#   - @on_event deprecated → contextmanager lifespan
#   - imports app.scraper en haut du fichier (plus d'import circulaire)
#   - /health sécurisé (ne crashe plus si modèles non chargés)
#   - /stats endpoint (monitoring des prédictions)
#   - /predict retourne 503 proprement si modèles non chargés

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import joblib, sys, os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

from src.preprocessing import clean_text
from app.scraper import scrape_trustpilot, scrape_amazon   # import en haut — plus de circulaire

# ── État global des modèles ───────────────────────────────────────────────────
vectorizer = None
modele     = None

# ── Compteurs de monitoring ───────────────────────────────────────────────────
_stats = {
    "predict_calls"      : 0,
    "predict_batch_calls": 0,
    "scrape_calls"       : 0,
    "started_at"         : None,
}

# ── Lifespan (remplace @on_event deprecated) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des modèles au démarrage, nettoyage à l'arrêt."""
    global vectorizer, modele
    models_path = os.path.join(BASE_PATH, 'models')
    try:
        vectorizer = joblib.load(os.path.join(models_path, 'tfidf_vectorizer.pkl'))
        modele     = joblib.load(os.path.join(models_path, 'best_model_champion.pkl'))
        _stats["started_at"] = datetime.now().isoformat()
        print(f"✅ Modèle    : {type(modele).__name__}")
        print(f"✅ Vectorizer: {vectorizer.get_params().get('max_features')} features TF-IDF")
    except Exception as e:
        print(f"❌ Erreur chargement modèles : {e}")
        print(f"   Chemin models/ attendu : {models_path}")
    yield
    # Teardown (optionnel — libérer mémoire)
    vectorizer = None
    modele     = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flipkart Sentiment Analysis API",
    description=(
        "Classification de sentiment pour reviews e-commerce.\n"
        "Classes : POSITIF / NEUTRE / NEGATIF\n"
        "Sources supportées : texte libre, Trustpilot, Amazon.in"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class ScrapeRequest(BaseModel):
    source: str            # "trustpilot" | "amazon"
    target: str            # nom entreprise ou ASIN
    max_pages: int = 3

# ── Fonction centrale de prédiction ──────────────────────────────────────────
def _check_models():
    """Lève 503 proprement si les modèles ne sont pas chargés."""
    if vectorizer is None or modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés. Vérifier que models/ contient les .pkl."
        )

def predict_one(text: str) -> dict:
    """Pipeline complet : texte brut → sentiment + probabilités."""
    cleaned    = clean_text(text)
    X          = vectorizer.transform([cleaned])
    label      = modele.predict(X)[0]
    probas     = modele.predict_proba(X)[0]
    classes    = list(modele.classes_)
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, probas)}
    return {
        "sentiment"    : label,
        "confidence"   : round(float(max(probas)), 4),
        "probabilities": proba_dict,
        "proba_POSITIF": proba_dict.get("POSITIF", 0),
        "proba_NEUTRE" : proba_dict.get("NEUTRE", 0),
        "proba_NEGATIF": proba_dict.get("NEGATIF", 0),
    }

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name"     : "Flipkart Sentiment Analysis API",
        "version"  : "2.1.0",
        "status"   : "running" if modele is not None else "models_not_loaded",
        "endpoints": ["/health", "/predict", "/predict/batch", "/scrape", "/stats", "/docs"],
    }

@app.get("/health")
def health():
    """Statut de l'API — ne crashe jamais même si modèles non chargés."""
    if modele is None or vectorizer is None:
        return {
            "status" : "degraded",
            "reason" : "Modèles non chargés",
            "tip"    : "Lancer depuis la RACINE : uvicorn app.main:app --reload"
        }
    return {
        "status"        : "ok",
        "model"         : type(modele).__name__,
        "tfidf_features": vectorizer.get_params().get("max_features"),
        "started_at"    : _stats["started_at"],
        "timestamp"     : datetime.now().isoformat(),
    }

@app.get("/stats")
def stats():
    """Compteurs de monitoring — nombre de prédictions depuis le démarrage."""
    total = _stats["predict_calls"] + _stats["predict_batch_calls"] + _stats["scrape_calls"]
    return {
        **_stats,
        "total_predictions": total,
    }

@app.post("/predict")
def predict(req: TextRequest):
    """Prédit le sentiment d'un texte unique."""
    _check_models()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
    _stats["predict_calls"] += 1
    result = predict_one(req.text)
    result["text_original"] = req.text[:200]
    return result

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Prédit le sentiment de jusqu'à 200 textes en une seule requête."""
    _check_models()
    if not req.texts:
        raise HTTPException(status_code=400, detail="La liste de textes est vide")
    if len(req.texts) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 textes par batch")

    _stats["predict_batch_calls"] += 1
    results = []
    for text in req.texts:
        pred = predict_one(text)
        pred["text_original"] = text[:100]
        results.append(pred)

    sentiments = [r["sentiment"] for r in results]
    n = len(results)
    stats_batch = {
        "total"              : n,
        "POSITIF"            : sentiments.count("POSITIF"),
        "NEUTRE"             : sentiments.count("NEUTRE"),
        "NEGATIF"            : sentiments.count("NEGATIF"),
        "pct_positif"        : round(sentiments.count("POSITIF") / n * 100, 1),
        "pct_negatif"        : round(sentiments.count("NEGATIF") / n * 100, 1),
        "confidence_moyenne" : round(sum(r["confidence"] for r in results) / n, 4),
    }
    return {"statistics": stats_batch, "results": results}

@app.post("/scrape")
def scrape_and_analyze(req: ScrapeRequest):
    """
    Scrape les reviews d'un site et retourne l'analyse de sentiment.
    - source = "trustpilot" : target = nom d'entreprise (ex: "amazon.in")
    - source = "amazon"     : target = ASIN (ex: "B0CX8GR1GR")
    """
    _check_models()
    _stats["scrape_calls"] += 1

    source_clean = req.source.lower().strip()
    if source_clean == "trustpilot":
        raw_reviews = scrape_trustpilot(req.target, max_pages=req.max_pages)
    elif source_clean == "amazon":
        raw_reviews = scrape_amazon(req.target, max_pages=req.max_pages)
    else:
        raise HTTPException(status_code=400,
                            detail="source doit être 'trustpilot' ou 'amazon'")

    if not raw_reviews:
        raise HTTPException(status_code=404,
                            detail=f"Aucune review trouvée pour '{req.target}'")

    analyzed = []
    for review in raw_reviews:
        pred = predict_one(review["text"])
        analyzed.append({**review, **pred})

    sentiments = [r["sentiment"] for r in analyzed]
    n = len(analyzed)
    result_stats = {
        "total"              : n,
        "source"             : req.source,
        "target"             : req.target,
        "POSITIF"            : sentiments.count("POSITIF"),
        "NEUTRE"             : sentiments.count("NEUTRE"),
        "NEGATIF"            : sentiments.count("NEGATIF"),
        "pct_positif"        : round(sentiments.count("POSITIF") / n * 100, 1),
        "pct_negatif"        : round(sentiments.count("NEGATIF") / n * 100, 1),
        "confidence_moyenne" : round(sum(r["confidence"] for r in analyzed) / n, 4),
        "alerte"             : sentiments.count("NEGATIF") / n > 0.30,
        "timestamp"          : datetime.now().isoformat(),
    }
    return {"statistics": result_stats, "reviews": analyzed}
