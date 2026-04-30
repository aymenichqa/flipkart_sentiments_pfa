
# src/models.py — fonctions d'entraînement réutilisables
# Auteur : Aymen Ichqarrane

import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def charger_modele(nom_fichier: str):
    """Charge un modèle sérialisé depuis le dossier models/."""
    chemin = os.path.join(MODELS_DIR, nom_fichier)
    return joblib.load(chemin)

def charger_vectorizer():
    """Charge le TF-IDF vectorizer."""
    return charger_modele('tfidf_vectorizer.pkl')

def charger_champion():
    """Charge le meilleur modèle identifié en Phase 3."""
    return charger_modele('best_model_champion.pkl')

def predire(texte_vectorise, modele=None):
    """
    Prédit le sentiment d'un vecteur TF-IDF.
    Retourne (label, probabilités).
    """
    if modele is None:
        modele = charger_champion()
    
    label = modele.predict(texte_vectorise)[0]
    probas = modele.predict_proba(texte_vectorise)[0]
    classes = modele.classes_
    
    return label, dict(zip(classes, probas.round(4).tolist()))