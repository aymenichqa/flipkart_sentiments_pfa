#!/usr/bin/env python3
"""
fix_preprocessing.py — Corrige automatiquement src/preprocessing.py
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(BASE_DIR, "src", "preprocessing.py")

CONTENT = '''# src/preprocessing.py — fonctions NLP réutilisables
# Auteure : Ihssane Moutchou

import re
import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

# ── Téléchargement silencieux des ressources NLTK ─────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


# ═══════════════════════════════════════════════════════════════════════════════
# clean_text — Pour V1 (TF-IDF + Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Nettoie un avis client anglais : lowercase → regex → tokenize → stopwords → lemmatize."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def normalize_sentiment(label: str) -> str:
    """Normalise les labels : Positif → POSITIF, etc."""
    mapping = {'Positif': 'POSITIF', 'Négatif': 'NEGATIF', 'Neutre': 'NEUTRE'}
    return mapping.get(label, label)


# ═══════════════════════════════════════════════════════════════════════════════
# clean_text_v2 — Pour V2 (XLM-RoBERTa / BERT multilingue)
# ═══════════════════════════════════════════════════════════════════════════════

from src.darija_mapper import detect_darija, translate_darija

def clean_text_v2(text: str, handle_darija: bool = True) -> str:
    """
    Version améliorée pour V2 uniquement.
    Traduit la Darija en français sans détruire la ponctuation/accents.
    """
    if not text or not isinstance(text, str):
        return ""
    
    if handle_darija and detect_darija(text):
        text = translate_darija(text)
    
    text = re.sub(r'\\s+', ' ', text).strip()
    return text
'''

print(f"📝 Écriture de : {TARGET}")

# Écrire le fichier
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(CONTENT)

# Vérification
with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

if "def clean_text_v2" in content:
    print("✅ clean_text_v2 est bien présent dans le fichier !")
else:
    print("❌ Échec de l'écriture")

# Vider le cache Python
import shutil
cache_dir = os.path.join(BASE_DIR, "src", "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("🗑️  Cache Python vidé")

print("\n👉 Tu peux maintenant relancer : python test_models.py")