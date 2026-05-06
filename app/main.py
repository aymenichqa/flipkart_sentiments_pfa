# app/main.py — Backend FastAPI V3 (A/B Testing V1 vs V2)
# ══════════════════════════════════════════════════════════════════════════════
# Lancer depuis la RACINE :  uvicorn app.main:app --reload --port 8000
#
# Architecture A/B Testing :
#   V1 — TF-IDF + Logistic Regression  (rapide, anglais uniquement)
#   V2 — XLM-RoBERTa (HuggingFace)     (multilingue : FR / AR / Darija / EN)
#
# Les deux modèles sont chargés EN MÉMOIRE au démarrage (lifespan).
# L'utilisateur choisit la version via le paramètre model_version.
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

from src.preprocessing import clean_text
from app.scraper import scrape_jumia, scrape_marjane, scrape_gmaps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE TransformerPredictor — Robuste face aux sorties HuggingFace variables
# ══════════════════════════════════════════════════════════════════════════════
#
# PROBLÈME CLASSIQUE :
# La pipeline HuggingFace peut retourner :
#   Format A : [{"label": "Positive", "score": 0.97}]          (return_all_scores=False)
#   Format B : [[{"label": "Positive", "score": 0.97}, ...]]   (return_all_scores=True)
# Un code non robuste crashe sur le format inattendu.
# Cette classe gère les DEUX formats sans planter.

class TransformerPredictor:
    """
    Prédicteur multilingue basé sur XLM-RoBERTa (HuggingFace).
    Gère les deux formats de sortie possibles de la pipeline.
    Supporte : Français, Arabe, Darija (romanisé), Anglais, Espagnol...
    """

    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Mapping des labels HuggingFace → labels internes
    LABEL_MAP = {
        "Positive": "POSITIF", "positive": "POSITIF", "LABEL_2": "POSITIF",
        "Neutral" : "NEUTRE",  "neutral" : "NEUTRE",  "LABEL_1": "NEUTRE",
        "Negative": "NEGATIF", "negative": "NEGATIF", "LABEL_0": "NEGATIF",
    }
    CLASSES = ["POSITIF", "NEUTRE", "NEGATIF"]

    def __init__(self):
        self._pipeline = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"⏳ Chargement XLM-RoBERTa ({self.MODEL_NAME})...")
            # return_all_scores=True → on récupère les probas pour les 3 classes
            self._pipeline = hf_pipeline(
                task              = "sentiment-analysis",
                model             = self.MODEL_NAME,
                tokenizer         = self.MODEL_NAME,
                return_all_scores = True,   # Format B — plus riche
                truncation        = True,
                max_length        = 512,
                device            = -1,     # CPU (-1), GPU (0)
            )
            logger.info("✅ XLM-RoBERTa chargé")
        except ImportError:
            raise ImportError("pip install transformers torch sentencepiece")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement XLM-RoBERTa : {e}")

    def _parse_hf_output(self, raw_output) -> dict:
        """
        Parse robuste des sorties HuggingFace.
        Accepte Format A (liste de dicts) ET Format B (liste de listes de dicts).
        """
        proba_dict = {"POSITIF": 0.0, "NEUTRE": 0.0, "NEGATIF": 0.0}

        # Normaliser : on veut toujours une liste de dicts {"label": ..., "score": ...}
        items = raw_output
        if isinstance(raw_output, list) and len(raw_output) > 0:
            # Format B : [[{...}, {...}]] → prendre le premier élément
            if isinstance(raw_output[0], list):
                items = raw_output[0]
            # Format A : [{...}] (return_all_scores=False) → enrichir avec 0.0 pour les autres
            elif isinstance(raw_output[0], dict):
                items = raw_output

        for item in items:
            if not isinstance(item, dict):
                continue
            label  = self.LABEL_MAP.get(item.get("label", ""), "")
            score  = float(item.get("score", 0.0))
            if label in proba_dict:
                proba_dict[label] = round(score, 4)

        return proba_dict

    def predict(self, text: str) -> dict:
        """Prédit le sentiment d'un texte (toute langue supportée)."""
        if not text or not text.strip():
            return self._empty_result()

        try:
            raw = self._pipeline(text.strip()[:1024])
            proba_dict = self._parse_hf_output(raw)
        except Exception as e:
            logger.error(f"Erreur inférence Transformer : {e}")
            return self._empty_result()

        best  = max(proba_dict, key=proba_dict.get)
        return {
            "sentiment"    : best,
            "confidence"   : proba_dict[best],
            "probabilities": proba_dict,
            "modele"       : "XLM-RoBERTa V2",
            "version"      : "v2",
        }

    def predict_batch(self, texts: list) -> list:
        """Prédit en batch — une seule passe forward pour tous les textes."""
        if not texts:
            return []
        try:
            clean   = [t.strip()[:1024] if t else "" for t in texts]
            raw_all = self._pipeline(clean)
            results = []
            for raw in raw_all:
                pd   = self._parse_hf_output(raw)
                best = max(pd, key=pd.get)
                results.append({
                    "sentiment"    : best,
                    "confidence"   : pd[best],
                    "probabilities": pd,
                    "modele"       : "XLM-RoBERTa V2",
                    "version"      : "v2",
                })
            return results
        except Exception as e:
            logger.error(f"Erreur batch Transformer : {e}")
            return [self._empty_result() for _ in texts]

    @staticmethod
    def _empty_result() -> dict:
        return {
            "sentiment": "NEUTRE", "confidence": 0.0,
            "probabilities": {"POSITIF": 0.0, "NEUTRE": 1.0, "NEGATIF": 0.0},
            "modele": "XLM-RoBERTa V2", "version": "v2",
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE V1Predictor — TF-IDF + Logistic Regression (baseline)
# ══════════════════════════════════════════════════════════════════════════════

class V1Predictor:
    """Prédicteur classique V1 — TF-IDF + LR. Anglais uniquement."""

    def __init__(self):
        models_dir   = os.path.join(BASE_PATH, "models")
        self._vec    = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        self._model  = joblib.load(os.path.join(models_dir, "best_model_champion.pkl"))
        logger.info(f"✅ V1 chargé : {type(self._model).__name__}")

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        X       = self._vec.transform([cleaned])
        label   = self._model.predict(X)[0]
        probas  = self._model.predict_proba(X)[0]
        classes = list(self._model.classes_)
        pd      = {c: round(float(p), 4) for c, p in zip(classes, probas)}
        for cls in ["POSITIF", "NEUTRE", "NEGATIF"]:
            pd.setdefault(cls, 0.0)
        return {
            "sentiment"    : label,
            "confidence"   : round(float(max(probas)), 4),
            "probabilities": pd,
            "modele"       : "TF-IDF + LR (V1)",
            "version"      : "v1",
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
# STATE GLOBAL & LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

predictor_v1: Optional[V1Predictor]         = None
predictor_v2: Optional[TransformerPredictor] = None

_stats = {
    "v1_calls"    : 0,
    "v2_calls"    : 0,
    "scrape_calls": 0,
    "started_at"  : None,
    "v1_disponible": False,
    "v2_disponible": False,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge V1 et V2 en mémoire au démarrage.
    Si V2 échoue (transformers non installé), l'API continue avec V1 uniquement.
    """
    global predictor_v1, predictor_v2

    _stats["started_at"] = datetime.now().isoformat()

    # ── Charger V1 (obligatoire) ──────────────────────────────────────────────
    try:
        predictor_v1         = V1Predictor()
        _stats["v1_disponible"] = True
        logger.info("✅ V1 (TF-IDF) prêt")
    except Exception as e:
        logger.error(f"❌ V1 non chargé : {e}")

    # ── Charger V2 (optionnel — peut échouer si transformers absent) ──────────
    try:
        predictor_v2            = TransformerPredictor()
        _stats["v2_disponible"] = True
        logger.info("✅ V2 (XLM-RoBERTa) prêt")
    except Exception as e:
        logger.warning(f"⚠️  V2 non chargé : {e}")
        logger.warning("    pip install transformers torch sentencepiece")

    yield  # L'API tourne ici

    # Teardown
    predictor_v1 = None
    predictor_v2 = None


# ══════════════════════════════════════════════════════════════════════════════
# APP FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Plateforme Sentiment E-commerce Maroc",
    description=(
        "**A/B Testing IA** : Compare TF-IDF (V1) vs XLM-RoBERTa (V2)\n\n"
        "**Cibles** : Jumia Maroc · Marjane Mall · Google Maps\n\n"
        "**Langues** : Français · Arabe · Darija · Anglais"
    ),
    version="3.0.0",
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
    text         : str
    model_version: Literal["v1", "v2"] = "v2"

class BatchRequest(BaseModel):
    texts        : List[str]
    model_version: Literal["v1", "v2"] = "v2"

class ScrapeRequest(BaseModel):
    source       : Literal["jumia", "marjane", "gmaps"]
    url          : str                           # URL cible
    model_version: Literal["v1", "v2"] = "v2"
    max_pages    : int = 2                       # pour Jumia et Marjane
    target_reviews: int = 20                     # pour Google Maps


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_predictor(version: str):
    """Retourne le bon prédicteur selon la version choisie."""
    if version == "v2":
        if predictor_v2 is None:
            raise HTTPException(
                503,
                "V2 (XLM-RoBERTa) non disponible. "
                "Installer : pip install transformers torch sentencepiece"
            )
        return predictor_v2
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
        "total"             : n,
        "POSITIF"           : sentiments.count("POSITIF"),
        "NEUTRE"            : sentiments.count("NEUTRE"),
        "NEGATIF"           : sentiments.count("NEGATIF"),
        "pct_positif"       : round(sentiments.count("POSITIF") / n * 100, 1),
        "pct_neutre"        : round(sentiments.count("NEUTRE")  / n * 100, 1),
        "pct_negatif"       : round(sentiments.count("NEGATIF") / n * 100, 1),
        "confidence_moyenne": round(sum(r["confidence"] for r in results) / n, 4),
        "alerte_negative"   : sentiments.count("NEGATIF") / n > 0.30,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "nom"          : "Plateforme Sentiment E-commerce Maroc",
        "version"      : "3.0.0",
        "v1_disponible": _stats["v1_disponible"],
        "v2_disponible": _stats["v2_disponible"],
        "endpoints"    : ["/health", "/predict", "/predict/batch", "/scrape", "/stats", "/docs"],
    }


@app.get("/health")
def health():
    """Statut de l'API et disponibilité des deux modèles."""
    return {
        "status"       : "ok",
        "started_at"   : _stats["started_at"],
        "timestamp"    : datetime.now().isoformat(),
        "modeles"      : {
            "v1": {
                "nom"       : "TF-IDF + Logistic Regression",
                "disponible": _stats["v1_disponible"],
                "langues"   : ["en"],
            },
            "v2": {
                "nom"       : "XLM-RoBERTa (HuggingFace)",
                "disponible": _stats["v2_disponible"],
                "langues"   : ["fr", "ar", "darija", "en", "es", "de", "..."],
            },
        },
    }


@app.get("/stats")
def stats():
    """Compteurs de prédictions depuis le démarrage."""
    total = _stats["v1_calls"] + _stats["v2_calls"] + _stats["scrape_calls"]
    return {**_stats, "total_predictions": total}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Prédit le sentiment d'un texte avec le modèle choisi.
    Inclut le nom du modèle utilisé dans la réponse (pour l'A/B Testing).
    """
    if not req.text.strip():
        raise HTTPException(400, "Le texte ne peut pas être vide.")

    predictor = _get_predictor(req.model_version)
    result    = predictor.predict(req.text)

    if req.model_version == "v1":
        _stats["v1_calls"] += 1
    else:
        _stats["v2_calls"] += 1

    result["text_original"]  = req.text[:300]
    result["model_version"]  = req.model_version
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Prédit le sentiment de jusqu'à 200 textes en batch."""
    if not req.texts:
        raise HTTPException(400, "Liste de textes vide.")
    if len(req.texts) > 200:
        raise HTTPException(400, "Maximum 200 textes par batch.")

    predictor = _get_predictor(req.model_version)
    results   = predictor.predict_batch(req.texts)

    for r, t in zip(results, req.texts):
        r["text_original"] = t[:100]
        r["model_version"] = req.model_version

    if req.model_version == "v1":
        _stats["v1_calls"] += len(results)
    else:
        _stats["v2_calls"] += len(results)

    return {
        "model_version": req.model_version,
        "statistics"   : _build_stats_summary(results),
        "results"      : results,
    }


@app.post("/scrape")
def scrape_and_analyze(req: ScrapeRequest):
    """
    Scrape les avis depuis une source marocaine et applique l'IA choisie.

    Sources :
      jumia   → URL page produit Jumia Maroc
      marjane → URL page produit Marjane Mall
      gmaps   → URL Google Maps du commerce local
    """
    predictor = _get_predictor(req.model_version)
    _stats["scrape_calls"] += 1

    # ── Scraping ──────────────────────────────────────────────────────────────
    print(f"\n🚀 Scraping {req.source} — {req.url}")
    try:
        if req.source == "jumia":
            raw_reviews = scrape_jumia(req.url, req.max_pages)
        elif req.source == "marjane":
            raw_reviews = scrape_marjane(req.url, req.max_pages)
        elif req.source == "gmaps":
            raw_reviews = scrape_gmaps(req.url, req.target_reviews)
        else:
            raise HTTPException(400, f"Source inconnue : {req.source}")
    except Exception as e:
        raise HTTPException(500, f"Erreur scraping : {e}")

    if not raw_reviews:
        raise HTTPException(404, f"Aucun avis trouvé sur {req.url}")

    # ── Prédiction batch (efficace pour Transformer) ──────────────────────────
    texts   = [r["text"] for r in raw_reviews]
    preds   = predictor.predict_batch(texts)

    analyzed = []
    for review, pred in zip(raw_reviews, preds):
        analyzed.append({
            **review,
            "sentiment"    : pred["sentiment"],
            "confidence"   : pred["confidence"],
            "probabilities": pred["probabilities"],
            "modele"       : pred["modele"],
            "model_version": req.model_version,
        })

    if req.model_version == "v1":
        _stats["v1_calls"] += len(analyzed)
    else:
        _stats["v2_calls"] += len(analyzed)

    stats_summary = {
        **_build_stats_summary(analyzed),
        "source"       : req.source,
        "url"          : req.url,
        "model_version": req.model_version,
        "modele_nom"   : preds[0]["modele"] if preds else "",
        "timestamp"    : datetime.now().isoformat(),
    }

    return {"statistics": stats_summary, "reviews": analyzed}