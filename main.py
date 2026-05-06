#!/usr/bin/env python3
"""
main.py — Point d'entrée du projet Flipkart Sentiment Analysis
==============================================================
Ce fichier racine permet de lancer les composants depuis la racine du projet.

Usage :
    python main.py api         → Lance l'API FastAPI (port 8000)
    python main.py dashboard   → Lance le dashboard Streamlit (port 8501)
    python main.py test        → Lance les tests de l'API
    python main.py predict "texte"  → Prédiction rapide en CLI
"""

import sys
import os
import subprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def run_api():
    """Lance l'API FastAPI avec uvicorn."""
    print("🚀 Démarrage de l'API sur http://localhost:8000")
    print("   Swagger UI disponible sur http://localhost:8000/docs\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        cwd=BASE_PATH
    )


def run_dashboard():
    """Lance le dashboard Streamlit."""
    print("🖥️  Démarrage du dashboard sur http://localhost:8501")
    print("   ⚠️  L'API doit tourner sur localhost:8000 en parallèle\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"],
        cwd=BASE_PATH
    )


def run_tests():
    """Lance les tests de l'API."""
    print("🧪 Lancement des tests...\n")
    subprocess.run([sys.executable, "test_api.py"], cwd=BASE_PATH)


def predict_cli(text: str):
    """Prédiction rapide depuis la ligne de commande, sans l'API."""
    sys.path.insert(0, BASE_PATH)
    import joblib
    from src.preprocessing import clean_text

    print(f"\n🔍 Analyse : \"{text}\"\n")

    vectorizer = joblib.load(os.path.join(BASE_PATH, 'models', 'tfidf_vectorizer.pkl'))
    modele     = joblib.load(os.path.join(BASE_PATH, 'models', 'best_model_champion.pkl'))

    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    label   = modele.predict(X)[0]
    probas  = modele.predict_proba(X)[0]
    classes = list(modele.classes_)

    icons = {"POSITIF": "🟢", "NEUTRE": "🟡", "NEGATIF": "🔴"}
    print(f"  Résultat  : {icons.get(label, '⚪')} {label}")
    print(f"  Confiance : {max(probas):.1%}")
    print(f"  Détail    :")
    for c, p in sorted(zip(classes, probas), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        print(f"    {c:10} {bar:20} {p:.1%}")


def show_help():
    print(__doc__)
    print("Exemples :")
    print('  python main.py api')
    print('  python main.py predict "This product is great!"')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "api":
        run_api()
    elif cmd == "dashboard":
        run_dashboard()
    elif cmd == "test":
        run_tests()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage : python main.py predict \"votre texte ici\"")
            sys.exit(1)
        predict_cli(sys.argv[2])
    else:
        print(f"Commande inconnue : '{cmd}'")
        show_help()
        sys.exit(1)