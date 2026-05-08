#!/usr/bin/env python3
"""
test_api.py — Suite de tests complète API V3.1 (Multi-Modèles XLM-RoBERTa)
═══════════════════════════════════════════════════════════════════════════
Lancer APRÈS avoir démarré l'API :
    uvicorn app.main:app --reload --port 8000
Puis :
    python test_api.py
═══════════════════════════════════════════════════════════════════════════
"""

import requests
import json
import time
from typing import List, Tuple

BASE_URL = "http://localhost:8000"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Phrases de test stratégiques (FR + Darija + Sarcasme + Emojis)
TEST_PHRASES: List[Tuple[str, str, str]] = [
    # (catégorie, langue, texte)
    ("positif_clair", "FR", "Ce produit est vraiment excellent ! Livraison rapide, je recommande."),
    ("positif_darija", "DAR", "Mzyan bzzaf had l'produit, srite men Jumia w wasalni f 2 jours !"),
    ("negatif_clair", "FR", "Très déçu, qualité médiocre et service client inexistant."),
    ("negatif_darija", "DAR", "Khayb had l'produit, mafhemch chno dert b had l flous, 7ram."),
    ("sarcasme", "FR", "Oh super, encore un chargeur qui marche 2 jours. Quelle qualité !"),
    ("emoji_social", "FR", "😍😍 j'adore vraiment trop bien ce téléphone omg"),
    ("neutre", "FR", "Le produit est arrivé dans les délais prévus. Rien à signaler."),
    ("darija_mixte", "DAR", "Wa3er l'produit mais l prix bzzaf, ma3endich flous bzzaf."),
]

# Modèles V2 à comparer
V2_MODELS = ["default", "twitter", "french"]

# URL Jumia de test (remplace par un vrai produit avec avis)
JUMIA_TEST_URL = "https://www.jumia.ma/catalog/productratingsreviews/sku/DE646EA0IFGGGNAFAMZ/"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ok(label: str, data=None):
    print(f"  ✅ {label}")
    if data:
        print(f"     {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")

def _fail(label: str, reason: str):
    print(f"  ❌ {label} → {reason}")

def _warn(label: str, reason: str):
    print(f"  ⚠️  {label} → {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests individuels
# ═══════════════════════════════════════════════════════════════════════════════

def test_health() -> bool:
    """GET /health — vérifie l'état des modèles."""
    print("\n🔍 [1/7] GET /health")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            v2_models = data["modeles"]["v2"].get("models_charges", [])
            _ok("Health OK", {
                "v1": data["modeles"]["v1"]["disponible"],
                "v2": data["modeles"]["v2"]["disponible"],
                "v2_models_charges": v2_models,
            })
            if len(v2_models) < len(V2_MODELS):
                _warn("Health", f"Seulement {len(v2_models)}/{len(V2_MODELS)} modèles V2 chargés")
            return True
        _fail("health", f"HTTP {r.status_code}")
        return False
    except Exception as e:
        _fail("health", str(e))
        return False


def test_predict_v1() -> bool:
    """POST /predict — V1 (TF-IDF, anglais uniquement)."""
    print("\n🔍 [2/7] POST /predict — V1 (TF-IDF)")
    payload = {
        "text": "This product is absolutely amazing! Great quality and fast delivery.",
        "model_version": "v1",
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        if r.status_code == 200:
            d = r.json()
            _ok(f"V1 → {d['sentiment']} ({d['confidence']:.2%})", {
                "modele": d.get("modele"),
                "probabilities": d.get("probabilities"),
            })
            return True
        _fail("predict v1", f"HTTP {r.status_code} — {r.text[:200]}")
        return False
    except Exception as e:
        _fail("predict v1", str(e))
        return False


def test_predict_v2_comparison() -> bool:
    """
    POST /predict — Compare les 3 modèles V2 sur les mêmes phrases.
    C'est LE test le plus important pour choisir ton modèle.
    """
    print("\n🔍 [3/7] POST /predict — Comparaison V2 (default vs twitter vs french)")
    print("     " + "─" * 70)
    print(f"     {'Phrase':<35} {'default':<12} {'twitter':<12} {'french':<12}")
    print("     " + "─" * 70)

    all_ok = True
    for category, lang, text in TEST_PHRASES:
        results = {}
        for model_key in V2_MODELS:
            payload = {
                "text": text,
                "model_version": "v2",
                "v2_model_key": model_key,
            }
            try:
                r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
                if r.status_code == 200:
                    d = r.json()
                    results[model_key] = (d["sentiment"], d["confidence"])
                elif r.status_code == 503:
                    _warn(f"V2-{model_key}", "Modèle non disponible (503)")
                    results[model_key] = ("N/A", 0.0)
                else:
                    _fail(f"V2-{model_key}", f"HTTP {r.status_code}")
                    results[model_key] = ("ERR", 0.0)
                    all_ok = False
            except Exception as e:
                _fail(f"V2-{model_key}", str(e))
                results[model_key] = ("ERR", 0.0)
                all_ok = False

        # Affichage aligné
        short_text = text[:32] + "..." if len(text) > 35 else text
        row = f"     [{lang}] {short_text:<28}"
        for mk in V2_MODELS:
            sentiment, conf = results.get(mk, ("?", 0))
            row += f" {sentiment:<6}({conf:.0%})  "
        print(row)

    print("     " + "─" * 70)
    print("     💡 Conseil : Si 'twitter' gagne sur la Darija, utilise-le pour Jumia/Twitter.")
    print("     💡 Conseil : Si 'default' gagne sur les avis longs, garde-le pour Trustpilot.")
    return all_ok


def test_predict_v2_single() -> bool:
    """Test V2 avec un modèle spécifique (default)."""
    print("\n🔍 [4/7] POST /predict — V2 spécifique (v2_model_key='twitter')")
    payload = {
        "text": "Wa3er l'produit, j'adore bzzaf ! ❤️",
        "model_version": "v2",
        "v2_model_key": "twitter",
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json()
            _ok(f"V2-twitter → {d['sentiment']} ({d['confidence']:.2%})", {
                "modele_utilise": d.get("modele"),
                "v2_model_key": d.get("v2_model_key"),
            })
            return True
        elif r.status_code == 503:
            _warn("V2-twitter", "Non disponible (503)")
            return True
        _fail("predict v2 twitter", f"HTTP {r.status_code}")
        return False
    except Exception as e:
        _fail("predict v2 twitter", str(e))
        return False


def test_predict_batch_v1() -> bool:
    """POST /predict/batch — V1."""
    print("\n🔍 [5/7] POST /predict/batch (V1)")
    payload = {
        "texts": [
            "Excellent product, very satisfied!",
            "Average quality, nothing special.",
            "Terrible! Broke after 2 days.",
            "Good value for the price.",
        ],
        "model_version": "v1",
    }
    try:
        r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=30)
        if r.status_code == 200:
            d = r.json()
            stats = d.get("statistics", {})
            _ok(f"Batch V1 OK — {stats.get('total', 0)} avis", {
                "POS": stats.get("POSITIF"),
                "NEU": stats.get("NEUTRE"),
                "NEG": stats.get("NEGATIF"),
                "confiance_moy": stats.get("confidence_moyenne"),
            })
            return True
        _fail("batch v1", f"HTTP {r.status_code}")
        return False
    except Exception as e:
        _fail("batch v1", str(e))
        return False


def test_predict_batch_v2() -> bool:
    """POST /predict/batch — V2 avec modèle spécifique."""
    print("\n🔍 [6/7] POST /predict/batch (V2 — french)")
    payload = {
        "texts": [
            "Ce produit est magnifique, je suis ravi !",
            "Qualité acceptable mais rien d'exceptionnel.",
            "Horrible expérience, je veux un remboursement.",
        ],
        "model_version": "v2",
        "v2_model_key": "french",
    }
    try:
        r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json()
            stats = d.get("statistics", {})
            _ok(f"Batch V2-french OK — {stats.get('total', 0)} avis", {
                "POS": stats.get("POSITIF"),
                "NEU": stats.get("NEUTRE"),
                "NEG": stats.get("NEGATIF"),
                "modele": d.get("v2_model_key"),
            })
            return True
        elif r.status_code == 503:
            _warn("batch v2", "V2 non disponible (503)")
            return True
        _fail("batch v2", f"HTTP {r.status_code}")
        return False
    except Exception as e:
        _fail("batch v2", str(e))
        return False


def test_scrape_jumia() -> bool:
    """POST /scrape — Jumia avec choix du modèle V2."""
    print("\n🔍 [7/7] POST /scrape — Jumia (V2-twitter pour Darija potentielle)")
    payload = {
        "source": "jumia",
        "url": JUMIA_TEST_URL,
        "model_version": "v2",
        "v2_model_key": "twitter",  # ← Meilleur pour les commentaires courts/darija
        "max_pages": 1,
    }
    try:
        r = requests.post(f"{BASE_URL}/scrape", json=payload, timeout=240)
        if r.status_code == 200:
            d = r.json()
            stats = d.get("statistics", {})
            _ok(f"Scraping OK — {stats.get('total', 0)} avis analysés", {
                "source": stats.get("source"),
                "POS": stats.get("POSITIF"),
                "NEG": stats.get("NEGATIF"),
                "modele": stats.get("modele_nom"),
                "v2_model_key": stats.get("v2_model_key"),
            })
            return True
        elif r.status_code == 404:
            _warn("scrape", "Aucun avis trouvé (404) — change l'URL produit")
            return True
        elif r.status_code == 503:
            _warn("scrape", "V2 non disponible (503)")
            return True
        else:
            _fail("scrape jumia", f"HTTP {r.status_code} — {r.text[:300]}")
            return False
    except Exception as e:
        _fail("scrape jumia", str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Runner principal
# ═══════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("Health Check", test_health),
    ("Predict V1", test_predict_v1),
    ("Comparaison V2 (A/B)", test_predict_v2_comparison),
    ("Predict V2 spécifique", test_predict_v2_single),
    ("Batch V1", test_predict_batch_v1),
    ("Batch V2", test_predict_batch_v2),
    ("Scrape Jumia V2", test_scrape_jumia),
]


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  SentimentLab — Test Suite V3.1 (Multi-Modèles XLM-RoBERTa)      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  API cible : {BASE_URL}")
    print("  Vérification de l'API...")
    time.sleep(1)

    # Vérification rapide que l'API est up
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except:
        print("\n  ❌ L'API ne répond pas sur {BASE_URL}")
        print("     Lance d'abord : uvicorn app.main:app --reload --port 8000")
        return

    results = []
    for name, fn in TESTS:
        print(f"\n{'─' * 70}")
        try:
            ok = fn()
        except Exception as exc:
            print(f"  💥 Exception inattendue : {exc}")
            ok = False
        results.append((name, ok))

    # Résumé
    print(f"\n{'═' * 70}")
    print("  RÉSUMÉ DES TESTS")
    print(f"{'═' * 70}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  Score : {passed}/{len(results)} tests passés")

    if passed == len(results):
        print("\n  🎉 Tous les tests passent — Multi-modèles V2 opérationnels !")
    else:
        print("\n  ⚠️  Certains tests échouent — vérifie les logs ci-dessus.")


if __name__ == "__main__":
    main()