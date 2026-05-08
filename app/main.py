# app/main.py — Backend FastAPI V3.1 (Multi-Modèles XLM-RoBERTa + Darija)
# ══════════════════════════════════════════════════════════════════════════════
# Lancer depuis la RACINE :  uvicorn app.main:app --reload --port 8000
#
# Architecture :
#   V1 — TF-IDF + Logistic Regression  (rapide, anglais uniquement)
#   V2 — XLM-RoBERTa (multi-variantes)  (multilingue : FR / AR / Darija / EN)
#
# Modèles V2 disponibles :
#   • "default"  → nlptown/bert-base-multilingual-uncased-sentiment (généraliste)
#   • "twitter"  → cardiffnlp/twitter-xlm-roberta-base-sentiment (réseaux sociaux)
#   • "french"   → philschmid/distilbert-base-multilingual-cased-sentiment-2
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import joblib
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Chemin racine du projet ───────────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

# ⬇️ CORRECTION : importe clean_text (V1, anglais) ET clean_text_v2 (V2, Darija)
from src.preprocessing import clean_text, clean_text_v2
from app.scraper import scrape_jumia, scrape_gmaps, scrape_trustpilot, scrape_tripadvisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE TransformerPredictor — V2 (XLM-RoBERTa / BERT multilingue)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerPredictor:
    """
    Prédicteur multilingue basé sur XLM-RoBERTa / BERT multilingue.
    Traduit automatiquement la Darija en français avant analyse.
    """

    AVAILABLE_MODELS = {
        "default": {
            "name": "nlptown/bert-base-multilingual-uncased-sentiment",
            "description": "Généraliste - E-commerce & avis longs",
            "label_map": {
                "5 stars": "POSITIF", "4 stars": "POSITIF",
                "3 stars": "NEUTRE",
                "2 stars": "NEGATIF", "1 star": "NEGATIF",
                "POSITIVE": "POSITIF", "positive": "POSITIF",
                "NEUTRAL": "NEUTRE", "neutral": "NEUTRE",
                "NEGATIVE": "NEGATIF", "negative": "NEGATIF",
                "LABEL_0": "NEGATIF", "LABEL_1": "NEUTRE", "LABEL_2": "POSITIF",
                "LABEL_3": "POSITIF", "LABEL_4": "POSITIF",
            }
        },
        "twitter": {
            "name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "description": "Textes courts / style réseaux sociaux / slang",
            "label_map": {
                "Positive": "POSITIF", "positive": "POSITIF", "LABEL_2": "POSITIF",
                "Neutral" : "NEUTRE",  "neutral" : "NEUTRE",  "LABEL_1": "NEUTRE",
                "Negative": "NEGATIF", "negative": "NEGATIF", "LABEL_0": "NEGATIF",
            }
        },
        "french": {
            "name": "philschmid/distilbert-base-multilingual-cased-sentiment-2",
            "description": "Optimisé pour le français standard",
            "label_map": {
                "positive": "POSITIF", "POSITIVE": "POSITIF",
                "neutral" : "NEUTRE",  "NEUTRAL": "NEUTRE",
                "negative": "NEGATIF", "NEGATIVE": "NEGATIF",
                "LABEL_0": "NEGATIF", "LABEL_1": "NEUTRE", "LABEL_2": "POSITIF",
            }
        }
    }

    CLASSES = ["POSITIF", "NEUTRE", "NEGATIF"]

    def __init__(self, model_key: str = "default"):
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modèle '{model_key}' inconnu. Choix : {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key
        self.model_config = self.AVAILABLE_MODELS[model_key]
        self.MODEL_NAME = self.model_config["name"]
        self.LABEL_MAP = self.model_config["label_map"]
        self._pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"⏳ Chargement modèle V2 '{self.model_key}' ({self.MODEL_NAME})...")
            self._pipeline = hf_pipeline(
                task              = "sentiment-analysis",
                model             = self.MODEL_NAME,
                tokenizer         = self.MODEL_NAME,
                return_all_scores = True,
                truncation        = True,
                max_length        = 512,
                device            = -1,     # CPU (-1), GPU (0)
            )
            logger.info(f"✅ Modèle V2 '{self.model_key}' chargé")
        except ImportError:
            raise ImportError("pip install transformers torch sentencepiece")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement {self.MODEL_NAME} : {e}")

    def _parse_hf_output(self, raw_output) -> dict:
        proba_dict = {"POSITIF": 0.0, "NEUTRE": 0.0, "NEGATIF": 0.0}

        # Cas : dict unique
        if isinstance(raw_output, dict):
            label = self.LABEL_MAP.get(raw_output.get("label", ""), "")
            score = float(raw_output.get("score", 0.0))
            if label in proba_dict:
                proba_dict[label] = round(score, 4)
            return proba_dict

        # Normaliser en liste de dicts
        items = raw_output
        if isinstance(raw_output, list) and len(raw_output) > 0:
            if isinstance(raw_output[0], list):
                items = raw_output[0]
            elif isinstance(raw_output[0], dict):
                items = raw_output

        for item in items:
            if not isinstance(item, dict):
                continue
            label = self.LABEL_MAP.get(item.get("label", ""), "")
            score = float(item.get("score", 0.0))
            if label in proba_dict:
                proba_dict[label] = round(score, 4)

        return proba_dict

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_result()

        text = text.strip()

        # ⬇️ CORRECTION : traduit la Darija en français AVANT d'analyser
        from src.darija_mapper import detect_darija, translate_darija
        if detect_darija(text):
            original = text
            text = translate_darija(text)
            logger.info(f"🔄 Darija détectée : '{original[:60]}...' → '{text[:60]}...'")

        try:
            raw = self._pipeline(text[:1024])
            proba_dict = self._parse_hf_output(raw)
        except Exception as e:
            logger.error(f"Erreur inférence Transformer ({self.model_key}) : {e}")
            return self._empty_result()

        best = max(proba_dict, key=proba_dict.get)
        return {
            "sentiment"    : best,
            "confidence"   : proba_dict[best],
            "probabilities": proba_dict,
            "modele"       : f"XLM-RoBERTa V2 [{self.model_key}]",
            "version"      : "v2",
            "v2_model_key" : self.model_key,
        }

    def predict_batch(self, texts: list) -> list:
        if not texts:
            return []
        try:
            # ⬇️ CORRECTION : traduit la Darija en batch avant analyse
            from src.darija_mapper import detect_darija, translate_darija
            clean = []
            for t in texts:
                t = t.strip()[:1024] if t else ""
                if t and detect_darija(t):
                    t = translate_darija(t)
                clean.append(t)
            
            raw_all = self._pipeline(clean)
            results = []
            for raw in raw_all:
                pd = self._parse_hf_output(raw)
                best = max(pd, key=pd.get)
                results.append({
                    "sentiment"    : best,
                    "confidence"   : pd[best],
                    "probabilities": pd,
                    "modele"       : f"XLM-RoBERTa V2 [{self.model_key}]",
                    "version"      : "v2",
                    "v2_model_key" : self.model_key,
                })
            return results
        except Exception as e:
            logger.error(f"Erreur batch Transformer ({self.model_key}) : {e}")
            return [self._empty_result() for _ in texts]

    def _empty_result(self) -> dict:
        return {
            "sentiment": "NEUTRE", "confidence": 0.0,
            "probabilities": {"POSITIF": 0.0, "NEUTRE": 1.0, "NEGATIF": 0.0},
            "modele": f"XLM-RoBERTa V2 [{self.model_key}]",
            "version": "v2",
            "v2_model_key": self.model_key,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE V1Predictor — TF-IDF + Logistic Regression (Anglais uniquement)
# ══════════════════════════════════════════════════════════════════════════════

class V1Predictor:
    def __init__(self):
        models_dir = os.path.join(BASE_PATH, "models")
        self._vec = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        self._model = joblib.load(os.path.join(models_dir, "best_model_champion.pkl"))
        logger.info(f"✅ V1 chargé : {type(self._model).__name__}")

    def predict(self, text: str) -> dict:
        # ⬇️ CORRECTION : clean_text original (anglais), PAS de Darija pour V1
        cleaned = clean_text(text)
        X = self._vec.transform([cleaned])
        label = self._model.predict(X)[0]
        probas = self._model.predict_proba(X)[0]
        classes = list(self._model.classes_)
        pd = {c: round(float(p), 4) for c, p in zip(classes, probas)}
        for cls in ["POSITIF", "NEUTRE", "NEGATIF"]:
            pd.setdefault(cls, 0.0)
        return {
            "sentiment"    : label,
            "confidence"   : round(float(max(probas)), 4),
            "probabilities": pd,
            "modele"       : "TF-IDF + LR (V1)",
            "version"      : "v1",
            "v2_model_key" : None,
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
# STATE GLOBAL & LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

predictor_v1: Optional[V1Predictor] = None
predictor_v2_models: dict[str, TransformerPredictor] = {}

_stats = {
    "v1_calls": 0,
    "v2_calls": 0,
    "scrape_calls": 0,
    "started_at": None,
    "v1_disponible": False,
    "v2_disponible": False,
    "v2_models_available": [],
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_v1, predictor_v2_models

    _stats["started_at"] = datetime.now().isoformat()

    # ── Charger V1 ────────────────────────────────────────────────────────────
    try:
        predictor_v1 = V1Predictor()
        _stats["v1_disponible"] = True
        logger.info("✅ V1 (TF-IDF) prêt")
    except Exception as e:
        logger.error(f"❌ V1 non chargé : {e}")

    # ── Charger TOUS les modèles V2 ───────────────────────────────────────────
    try:
        for model_key in TransformerPredictor.AVAILABLE_MODELS.keys():
            try:
                logger.info(f"⏳ Chargement V2 modèle '{model_key}'...")
                predictor_v2_models[model_key] = TransformerPredictor(model_key)
                _stats["v2_models_available"].append(model_key)
                logger.info(f"✅ V2 modèle '{model_key}' prêt")
            except Exception as e:
                logger.warning(f"⚠️ V2 modèle '{model_key}' non chargé : {e}")
        
        if predictor_v2_models:
            _stats["v2_disponible"] = True
            logger.info(f"✅ {len(predictor_v2_models)} modèle(s) V2 chargé(s)")
    except Exception as e:
        logger.warning(f"⚠️ Aucun modèle V2 chargé : {e}")

    yield  # L'API tourne ici

    # Teardown propre
    predictor_v1 = None
    predictor_v2_models.clear()
    logger.info("🛑 Modèles déchargés")


# ══════════════════════════════════════════════════════════════════════════════
# APP FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Plateforme Sentiment E-commerce Maroc",
    description=(
        "**A/B Testing IA** : Compare TF-IDF (V1) vs XLM-RoBERTa (V2 multi-variantes)\n\n"
        "**V2 disponibles** : `default` (e-commerce), `twitter` (social/short), `french` (FR pur)\n\n"
        "**Cibles** : Jumia Maroc · Trustpilot · TripAdvisor · Google Maps\n\n"
        "**Langues** : Français · Arabe · Darija · Anglais"
    ),
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    model_version: Literal["v1", "v2"] = "v2"
    v2_model_key: Literal["default", "twitter", "french"] = "default"

class BatchRequest(BaseModel):
    texts: List[str]
    model_version: Literal["v1", "v2"] = "v2"
    v2_model_key: Literal["default", "twitter", "french"] = "default"

class ScrapeRequest(BaseModel):
    source: Literal["jumia", "gmaps", "trustpilot", "tripadvisor"]
    url: str
    model_version: Literal["v1", "v2"] = "v2"
    v2_model_key: Literal["default", "twitter", "french"] = "default"
    max_pages: int = 2
    target_reviews: int = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_predictor(version: str, v2_model_key: str = "default"):
    """
    Retourne le bon prédicteur.
    Pour V2, sélectionne le sous-modèle via v2_model_key.
    """
    if version == "v2":
        if not predictor_v2_models:
            raise HTTPException(
                503,
                "V2 (XLM-RoBERTa) non disponible. "
                "Installer : pip install transformers torch sentencepiece"
            )
        if v2_model_key not in predictor_v2_models:
            available = list(predictor_v2_models.keys())
            raise HTTPException(
                503,
                f"Modèle V2 '{v2_model_key}' non chargé. Disponibles : {available}"
            )
        return predictor_v2_models[v2_model_key]
    else:
        if predictor_v1 is None:
            raise HTTPException(503, "V1 (TF-IDF) non disponible.")
        return predictor_v1

def _build_stats_summary(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}
    sentiments = [r["sentiment"] for r in results]
    return {
        "total": n,
        "POSITIF": sentiments.count("POSITIF"),
        "NEUTRE": sentiments.count("NEUTRE"),
        "NEGATIF": sentiments.count("NEGATIF"),
        "pct_positif": round(sentiments.count("POSITIF") / n * 100, 1),
        "pct_neutre": round(sentiments.count("NEUTRE") / n * 100, 1),
        "pct_negatif": round(sentiments.count("NEGATIF") / n * 100, 1),
        "confidence_moyenne": round(sum(r["confidence"] for r in results) / n, 4),
        "alerte_negative": sentiments.count("NEGATIF") / n > 0.30,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "nom": "Plateforme Sentiment E-commerce Maroc",
        "version": "3.1.0",
        "v1_disponible": _stats["v1_disponible"],
        "v2_disponible": _stats["v2_disponible"],
        "v2_models_disponibles": _stats["v2_models_available"],
        "endpoints": ["/health", "/predict", "/predict/batch", "/scrape", "/stats", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "started_at": _stats["started_at"],
        "timestamp": datetime.now().isoformat(),
        "modeles": {
            "v1": {
                "nom": "TF-IDF + Logistic Regression",
                "disponible": _stats["v1_disponible"],
                "langues": ["en"],
            },
            "v2": {
                "nom": "XLM-RoBERTa / BERT multilingue",
                "disponible": _stats["v2_disponible"],
                "models_charges": _stats["v2_models_available"],
                "langues": ["fr", "ar", "darija", "en", "es", "de"],
            },
        },
    }


@app.get("/stats")
def stats():
    total = _stats["v1_calls"] + _stats["v2_calls"] + _stats["scrape_calls"]
    return {**_stats, "total_predictions": total}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "Le texte ne peut pas être vide.")

    predictor = _get_predictor(req.model_version, req.v2_model_key)
    result = predictor.predict(req.text)

    if req.model_version == "v1":
        _stats["v1_calls"] += 1
    else:
        _stats["v2_calls"] += 1

    result["text_original"] = req.text[:300]
    result["model_version"] = req.model_version
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(400, "Liste de textes vide.")
    if len(req.texts) > 200:
        raise HTTPException(400, "Maximum 200 textes par batch.")

    predictor = _get_predictor(req.model_version, req.v2_model_key)
    results = predictor.predict_batch(req.texts)

    for r, t in zip(results, req.texts):
        r["text_original"] = t[:100]
        r["model_version"] = req.model_version

    if req.model_version == "v1":
        _stats["v1_calls"] += len(results)
    else:
        _stats["v2_calls"] += len(results)

    return {
        "model_version": req.model_version,
        "v2_model_key": req.v2_model_key if req.model_version == "v2" else None,
        "statistics": _build_stats_summary(results),
        "results": results,
    }


@app.post("/scrape")
def scrape_and_analyze(req: ScrapeRequest):
    predictor = _get_predictor(req.model_version, req.v2_model_key)
    _stats["scrape_calls"] += 1

    print(f"\n🚀 Scraping {req.source} — {req.url} (modèle: {req.model_version}/{req.v2_model_key})")
    try:
        if req.source == "jumia":
            raw_reviews = scrape_jumia(req.url, req.max_pages)
        elif req.source == "gmaps":
            raw_reviews = scrape_gmaps(req.url, req.target_reviews)
        elif req.source == "trustpilot":
            raw_reviews = scrape_trustpilot(req.url, req.max_pages)
        elif req.source == "tripadvisor":
            raw_reviews = scrape_tripadvisor(req.url, req.max_pages)
        else:
            raise HTTPException(400, f"Source inconnue : {req.source}")
    except Exception as e:
        raise HTTPException(500, f"Erreur scraping : {e}")

    if not raw_reviews:
        raise HTTPException(404, f"Aucun avis trouvé sur {req.url}")

    texts = [r["text"] for r in raw_reviews]
    preds = predictor.predict_batch(texts)

    analyzed = []
    for review, pred in zip(raw_reviews, preds):
        analyzed.append({
            **review,
            "sentiment": pred["sentiment"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "modele": pred["modele"],
            "model_version": req.model_version,
            "v2_model_key": req.v2_model_key if req.model_version == "v2" else None,
        })

    if req.model_version == "v1":
        _stats["v1_calls"] += len(analyzed)
    else:
        _stats["v2_calls"] += len(analyzed)

    stats_summary = {
        **_build_stats_summary(analyzed),
        "source": req.source,
        "url": req.url,
        "model_version": req.model_version,
        "v2_model_key": req.v2_model_key if req.model_version == "v2" else None,
        "modele_nom": preds[0]["modele"] if preds else "",
        "timestamp": datetime.now().isoformat(),
    }

    return {"statistics": stats_summary, "reviews": analyzed}