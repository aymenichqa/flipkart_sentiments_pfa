 # src/preprocessing.py — fonctions NLP réutilisables
# Auteure : Ihssane Moutchou 

import re, nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Nettoie un avis client : lowercase → regex → tokenize → stopwords → lemmatize."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in _stop_words and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def normalize_sentiment(label: str) -> str:
    """Normalise les labels : Positif → POSITIF, etc."""
    mapping = {'Positif':'POSITIF','Négatif':'NEGATIF','Neutre':'NEUTRE'}
    return mapping.get(label, label)
