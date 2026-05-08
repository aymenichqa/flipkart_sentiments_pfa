# SentimentLab — Plateforme d'Analyse de Sentiments E-commerce

> **PFA (Projet de Fin d'Année) — Filière Data Science & IA**
> *Analyse de sentiments multilingue pour avis e-commerce marocain (Français, Arabe, Darija, Anglais)*

---

## 📋 Table des Matières
- [Auteurs](#auteurs)
- [Contexte & Problématique](#contexte--problématique)
- [Stack Technique](#stack-technique)
- [Architecture du Projet](#architecture-du-projet)
- [Pipeline ML Complet](#pipeline-ml-complet)
  - [1. Dataset Flipkart](#1-dataset-flipkart)
  - [2. Analyse Exploratoire (EDA)](#2-analyse-exploratoire-eda)
  - [3. Prétraitement NLP](#3-prétraitement-nlp)
  - [4. Modèles V1 — Machine Learning Classique](#4-modèles-v1--machine-learning-classique)
  - [5. Modèle V2 — Deep Learning Multilingue](#5-modèle-v2--deep-learning-multilingue)
  - [6. Comparaison & Sélection du Champion](#6-comparaison--sélection-du-champion)
- [Fonctionnalités de l'Application](#fonctionnalités-de-lapplication)
- [Installation & Configuration](#installation--configuration)
- [Guide d'Utilisation](#guide-dutilisation)
- [API Endpoints](#api-endpoints)
- [Performances & Métriques](#performances--métriques)
- [Améliorations Futures](#améliorations-futures)
- [Annexe : Structure Détaillée](#annexe--structure-détaillée)

---

## 👥 Auteurs

| Nom | Rôle | Contributions |
|-----|------|---------------|
| **Ihssane Moutchou** | ML Engineer | Preprocessing NLP, Modèle SVM, Scrapers, Dashboard |
| **Aymen Ichqarrane** | ML Engineer | EDA, Modèles LR/NB, API FastAPI, Architecture logicielle |

---

## 🎯 Contexte & Problématique

### Problème Métier
Les e-commerçants marocains (Jumia, vendeurs locaux) reçoivent des milliers d'avis clients mélangeant **Français, Arabe, Darija (arabe marocain romanisé) et Anglais**. Analyser manuellement ces avis est impossible à grande échelle.

### Solution
**SentimentLab** est une plateforme complète qui :
1. **Collecte** automatiquement les avis via scraping multi-sources
2. **Analyse** le sentiment avec 2 pipelines ML (classique + deep learning)
3. **Traduit** la Darija en français automatiquement
4. **Restitue** les résultats via une API et un dashboard temps réel

---

## 🛠️ Stack Technique

| Catégorie | Technologies |
|-----------|--------------|
| **Langage** | Python 3.12 |
| **ML Classique (V1)** | scikit-learn 1.8, TF-IDF, Logistic Regression, SVM, Naive Bayes |
| **Deep Learning (V2)** | Hugging Face Transformers 5.8, XLM-RoBERTa, PyTorch 2.11 |
| **NLP** | NLTK (tokenisation, stopwords, lemmatisation), regex |
| **Scraping** | Playwright 1.59, Beautiful Soup 4.14, fake-useragent |
| **API** | FastAPI 0.136, Uvicorn 0.46, Pydantic 2.13 |
| **Dashboard** | Streamlit 1.57, Plotly 6.7, Pandas 3.0 |
| **Base de Données** | SQLite (historique des analyses) |
| **Visualisations** | Matplotlib 3.10, Seaborn 0.13, Plotly, WordCloud |
| **Outils** | Jupyter Notebooks, joblib, python-dotenv |

---

## 📁 Architecture du Projet

```
flipkart_sentiments_pfa/
│
├── main.py                   # CLI : python main.py [api|dashboard|test|predict]
├── requirements.txt          # 123 dépendances complètes
│
├── app/                      ◄── APPLICATION (couche présentation)
│   ├── main.py              # API FastAPI — 537 lignes, 3 modèles V2
│   ├── dashboard.py         # Dashboard Streamlit — 3 onglets
│   ├── scraper.py           # 4 scrapers : Jumia, Google Maps, Trustpilot, TripAdvisor
│   └── requirements.txt     # Dépendances minimales API
│
├── src/                      ◄── SOURCE ML (couche métier)
│   ├── preprocessing.py     # clean_text (V1) + clean_text_v2 (V2 via Darija)
│   ├── models.py            # Chargement modèles, prédiction, GridSearch
│   ├── evaluate.py          # Métriques : accuracy, F1-weighted, F1-macro, matrice confusion
│   └── darija_mapper.py     # 120+ mots Darija → Français, détection automatique
│
├── notebooks/               ◄── EXPÉRIMENTATION & RECHERCHE
│   ├── 01_EDA_Aymen.ipynb          # Analyse exploratoire des données Flipkart
│   ├── 02_preprocessing_Ihssane.ipynb  # Pipeline NLP complet
│   ├── 03_models_LR_NB_Aymen.ipynb     # Logistic Regression + Naive Bayes
│   ├── 03_models_SVM_Ihssane.ipynb     # SVM linéaire + GridSearch
│   ├── 04_comparison_FINAL.ipynb       # Benchmark & sélection champion
│   └── 05_scraping_pipeline.ipynb      # Test des scrapers
│
├── data/                     ◄── DONNÉES
│   ├── flipkart_data_preprocessed.csv      # 9 976 avis prétraités
│   ├── flipkart_data_with_sentiment.csv    # Dataset brut labellisé
│   ├── X_train.pkl / X_test.pkl            # Vecteurs TF-IDF
│   ├── y_train.csv / y_test.csv            # Labels train/test
│   └── history.db                          # SQLite — historique utilisateur
│
├── models/                   ◄── MODÈLES ENTRAÎNÉS
│   ├── tfidf_vectorizer.pkl        # Vectorizer TF-IDF (max_features=10 000)
│   ├── logistic_regression.pkl     # Logistic Regression (C=1, GridSearch)
│   ├── naive_bayes.pkl            # Multinomial Naive Bayes (alpha=0.5)
│   ├── svm_model.pkl              # LinearSVC (C=0.1, GridSearch)
│   ├── best_model_champion.pkl    # Champion : Logistic Regression
│   └── comparison_results.pkl     # Résultats du benchmark
│
├── visualisations/           ◄── 15+ GRAPHIQUES
│   ├── comparison_*.png             # Radar, heatmap, F1 par classe
│   ├── lr_*.png / nb_*.png         # Confusion matrix, feature importance
│   ├── svm_*.png                   # Learning curve, confidence distribution
│   └── scraping_analysis.png       # Stats scraping
│
├── test_api.py               # Tests API (endpoints /predict, /scrape)
├── test_models.py            # Tests unitaires modèles
├── build_scraping_and_app.py # Script build pour déploiement
├── fix_preprocessing.py      # Script de correction dataset
└── .env                      # Variables d'environnement (HF_TOKEN)
```

---

## 🔬 Pipeline ML Complet

### 1. Dataset Flipkart

| Propriété | Valeur |
|-----------|--------|
| **Source** | [Kaggle — Flipkart Product Reviews](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) |
| **Taille** | 9 976 avis |
| **Split** | 80/20 (7 980 train / 1 996 test) — stratifié par classe |
| **Seed** | `random_state=42` |

#### Distribution des classes

| Classe | Effectif | % | Règle |
|--------|----------|---|-------|
| **POSITIF** | 8 091 | 81.1% | rating ≥ 4★ |
| **NÉGATIF** | 1 001 | 10.0% | rating ≤ 2★ |
| **NEUTRE** | 884 | 8.9% | rating = 3★ |

> ⚠️ **Déséquilibre critique** : 81% POSITIF. Les classes NEUTRE et NÉGATIF sont sous-représentées, ce qui impacte directement les performances F1.

#### Features du dataset brut
- `review` : texte brut de l'avis (anglais principalement)
- `rating` : note de 1 à 5 étoiles
- `sentiment` : label dérivé (Positif/Neutre/Négatif)
- `review_length` / `word_count` : métriques textuelles
- `lang` : langue détectée (langdetect)

### 2. Analyse Exploratoire (EDA)

**Notebook :** `notebooks/01_EDA_Aymen.ipynb`

Analyses réalisées :
- Distribution des ratings et des sentiments
- Longueur des reviews par classe
- Nuages de mots (WordCloud) par sentiment
- N-grams les plus fréquents (positifs vs négatifs)
- Corrélation rating ↔ sentiment

**Insights clés :**
- Les avis positifs sont en moyenne plus longs que les négatifs
- Les bigrammes comme "very good", "highly recommend" dominent la classe POSITIF
- "not good", "very disappointed" sont discriminants pour NÉGATIF

### 3. Prétraitement NLP

**Notebook :** `notebooks/02_preprocessing_Ihssane.ipynb`  
**Module :** `src/preprocessing.py`

#### Pipeline V1 (anglais uniquement)
```
Raw text → lower() → regex [^a-z\s] → word_tokenize → stopwords → lemmatize → clean text
```

#### Pipeline V2 (multilingue avec Darija)
```
Raw text → detect_darija() → [si Darija: translate_darija()] → clean_text_v2 → tokenizer BERT
```

#### Vectorisation (V1)
`TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))`

### 4. Modèles V1 — Machine Learning Classique

**Notebooks :**
- `notebooks/03_models_LR_NB_Aymen.ipynb` — Logistic Regression + Naive Bayes
- `notebooks/03_models_SVM_Ihssane.ipynb` — SVM linéaire

#### Modèles entraînés

| Modèle | GridSearch | Meilleurs paramètres |
|--------|-----------|---------------------|
| **Logistic Regression** | `C: [0.01, 0.1, 1, 10]`, `solver: lbfgs` | C=1 |
| **Multinomial NB** | `alpha: [0.1, 0.5, 1.0]` | alpha=0.5 |
| **LinearSVC** | `C: [0.01, 0.1, 1, 10]`, `loss: hinge/squared_hinge` | C=0.1, loss=squared_hinge |

#### Résultats V1

| Modèle | Accuracy | F1-weighted | F1-macro |
|--------|----------|-------------|----------|
| **Logistic Regression 🏆** | **0.877** | **0.858** | **0.642** |
| SVM Linéaire | 0.874 | 0.855 | 0.632 |
| Naive Bayes | 0.847 | 0.825 | 0.568 |

#### Analyse des erreurs
- **POSITIF** : F1 ≈ 0.93 — Très bien détecté (classe majoritaire)
- **NÉGATIF** : F1 ≈ 0.69 — Confusion avec NEUTRE
- **NEUTRE** : F1 ≈ 0.29 — Très mal détecté (classe minoritaire, 8.9%)

### 5. Modèle V2 — Deep Learning Multilingue

**Module :** `app/main.py` — classe `TransformerPredictor`

#### 3 variantes disponibles

| Variante | Modèle HF | Optimisé pour | Label Map |
|----------|-----------|---------------|-----------|
| **Default** 🎯 | `nlptown/bert-base-multilingual-uncased-sentiment` | Avis e-commerce longs | 5★→POSITIF, 3★→NEUTRE, 1★→NÉGATIF |
| **Twitter** 🐦 | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Textes courts, emojis, slang | Positive→POSITIF, Neutral→NEUTRE, Negative→NÉGATIF |
| **French** 🥖 | `philschmid/distilbert-base-multilingual-cased-sentiment-2` | Français formel | positive→POSITIF, neutral→NEUTRE, negative→NÉGATIF |

#### Fonctionnement
1. **Détection Darija** : `detect_darija()` → 15% seuil de mots reconnus
2. **Traduction** : `translate_darija()` via dictionnaire de 120+ entrées
3. **Inférence** : pipeline Hugging Face avec `return_all_scores=True`
4. **Post-traitement** : mapping des labels vers POSITIF/NEUTRE/NÉGATIF

### 6. Comparaison & Sélection du Champion

**Notebook :** `notebooks/04_comparison_FINAL.ipynb`

#### Résultats consolidés

| Modèle | Accuracy | F1-weighted | F1-macro | Tps entraînement |
|--------|----------|-------------|----------|-----------------|
| **Logistic Regression 🏆** | **0.877** | **0.858** | **0.642** | 2.3s |
| SVM Linéaire | 0.874 | 0.855 | 0.632 | 1.8s |
| Naive Bayes | 0.847 | 0.825 | 0.568 | **0.4s** |
| XLM-RoBERTa (V2) | — | — | — | ~30s (inférence) |

> **Champion : Logistic Regression** — Meilleur compromis performance/vitesse pour la V1. La V2 (XLM-RoBERTa) est sélectionnée pour le multilingue (FR/AR/Darija).

---

## ✨ Fonctionnalités de l'Application

### 1. 🧪 Test en Direct (Tab 1)
- Analyse individuelle d'un avis
- Choix entre V1 (TF-IDF, rapide) et V2 (XLM-RoBERTa, multilingue)
- 3 variantes V2 : Default, Twitter, French
- **Mode Comparaison** : test simultané des 3 modèles V2 avec identification du meilleur
- Exemples prédéfinis (FR, Darija, Sarcasme, Émojis)
- Affichage : sentiment, confiance, barres de probabilités

### 2. 🕸️ Veille Scraping (Tab 2)
- **4 sources** : Jumia Maroc, Google Maps, Trustpilot, TripAdvisor
- Scraping via Playwright (navigation headless Chromium)
- Analyse automatique des avis scrapés
- KPIs : total, positifs, neutres, négatifs
- **Alerte** : détection de >30% d'avis négatifs
- Graphiques : donut sentiments, histogramme confiance
- Export tableau des avis analysés

### 3. 📁 Archives & Historique (Tab 3)
- Sauvegarde SQLite de toutes les analyses
- Statistiques globales (total, prédictions, scrapings, modèle favori)
- **Timeline** des 20 dernières analyses
- **Export CSV** de l'historique complet
- Migration automatique des données (correction NaN)

### 4. 🌐 API REST (FastAPI)

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info API, statut modèles |
| `/health` | GET | Health check avec détails modèles |
| `/stats` | GET | Statistiques d'utilisation |
| `/predict` | POST | Analyse d'un texte (V1 ou V2) |
| `/predict/batch` | POST | Analyse batch (max 200 textes) |
| `/scrape` | POST | Scraping + analyse (4 sources) |
| `/docs` | GET | Swagger UI interactive |

### 5. 🧠 Darija Mapper
- **120+ entrées** de Darija romanisée → Français
- Détection automatique (seuil 15%)
- Couvre : sentiments, verbes, pronoms, quantités, expressions e-commerce
- Exemple : `"Mzyan bzzaf had l'produit"` → `"Bon beaucoup ce le produit"`

---

## ⚙️ Installation & Configuration

### Prérequis
- Python 3.10+
- Git
- Chrome/Chromium (pour Playwright)

### Installation

```bash
# 1. Cloner
git clone <votre-repo>
cd flipkart_sentiments_pfa

# 2. Environnement virtuel
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Playwright (pour le scraping)
playwright install chromium
```

### Configuration (.env)
```env
# Optionnel : token Hugging Face (taux limit augmenté)
HF_TOKEN=hf_votre_token_ici
```

---

## 🚀 Guide d'Utilisation

### Méthode 1 : CLI (recommandée)

```bash
# Terminal 1 : API
python main.py api
# → http://localhost:8000

# Terminal 2 : Dashboard
python main.py dashboard
# → http://localhost:8501

# Prédiction rapide
python main.py predict "Ce produit est vraiment excellent !"
```

### Méthode 2 : Manuel

```bash
# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run app/dashboard.py
```

### Tests

```bash
python main.py test
# ou
python test_api.py
python test_models.py
```

---

## 📊 Performances & Métriques

### V1 — Logistic Regression (Champion)

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| POSITIF | 0.89 | 0.98 | 0.93 | 1 618 |
| NÉGATIF | 0.83 | 0.59 | 0.69 | 200 |
| NEUTRE | 0.56 | 0.20 | 0.29 | 178 |
| **Accuracy** | | | **0.877** | 1 996 |
| **F1-weighted** | | | **0.858** | |
| **F1-macro** | | | **0.642** | |

### V2 — XLM-RoBERTa (Multilingue)

| Variante | Forces | Faiblesses |
|----------|--------|------------|
| Default 🎯 | Avis longs et structurés | Lent au chargement |
| Twitter 🐦 | Darija, emojis, textes courts | Moins précis sur texte long |
| French 🥖 | Français formel | Limitée au français |

### Recommandations d'utilisation

| Type de texte | Modèle recommandé |
|---------------|-------------------|
| Anglais | V1 (rapide) ou Default V2 |
| Français | French V2 |
| Darija / Arabe | Twitter V2 |
| Mélange multilingue | Default V2 |
| Commentaires courts | Twitter V2 |
| Reviews longues | Default V2 |

---

## 🧪 Tests & Qualité

### Tests d'API
- `test_api.py` : Teste les endpoints `/predict`, `/predict/batch`, `/scrape`
- Vérifie les codes HTTP, la structure JSON, les timeouts

### Tests de modèles
- `test_models.py` : Charge chaque modèle et vectorizer
- Valide les prédictions et les probabilités

### Scraping
- `build_scraping_and_app.py` : Script de déploiement
- `fix_preprocessing.py` : Script de correction des données

---

## 🔮 Améliorations Futures

### Court terme
- [ ] **SMOTE / class_weight** pour équilibrer le dataset
- [ ] **Fine-tuning** XLM-RoBERTa sur données e-commerce marocaines
- [ ] **API key** rate limiting pour sécuriser l'API

### Moyen terme
- [ ] **Cache** des modèles V2 pour temps de chargement réduit
- [ ] **Authentification** pour le dashboard multi-utilisateur
- [ ] **Déploiement Docker** avec docker-compose

### Long terme
- [ ] **Modèle V3** : GPT fine-tuned ou LLM local (Llama, Mistral)
- [ ] **Dashboard temps réel** avec WebSockets
- [ ] **Base de données vectorielle** (FAISS) pour recherche sémantique
- [ ] **Pipeline CI/CD** GitHub Actions

---

## 📝 Notes Importantes

- **Premier lancement** : le téléchargement des modèles Hugging Face peut prendre 2-5 minutes
- **Connexion internet** nécessaire pour V2 (poids des modèles ~500 Mo chacun)
- **RAM** : minimum 4 Go recommandé (8 Go pour les 3 modèles V2 simultanément)
- **Playwright** : nécessite Chromium installé (`playwright install chromium`)
- **Dashboard** : l'API doit tourner **avant** le dashboard (port 8000)
- **Historique** : fichier `data/history.db` créé automatiquement au premier lancement

---

## 📚 Annexe : Structure Détaillée

### Fichiers source (`src/`)

| Fichier | Rôle | Auteur |
|---------|------|--------|
| `preprocessing.py` | Nettoyage texte V1 + V2 | Ihssane |
| `models.py` | Chargement, prédiction, GridSearch | Aymen |
| `evaluate.py` | Métriques, rapport classification, tableau comparatif | Aymen & Ihssane |
| `darija_mapper.py` | Détection et traduction Darija → Français | Ihssane |

### Application (`app/`)

| Fichier | Rôle | Lignes |
|---------|------|--------|
| `main.py` | API FastAPI (V1 + 3 variantes V2) | 537 |
| `dashboard.py` | Dashboard Streamlit (3 onglets) | 620+ |
| `scraper.py` | 4 scrapers Playwright | ~400 |

### Modèles (`models/`)

| Fichier | Type | Taille |
|---------|------|--------|
| `tfidf_vectorizer.pkl` | Vectorizer TF-IDF | ~2 Mo |
| `logistic_regression.pkl` | Logistic Regression | ~500 Ko |
| `naive_bayes.pkl` | Multinomial NB | ~500 Ko |
| `svm_model.pkl` | LinearSVC | ~500 Ko |
| `best_model_champion.pkl` | LR (champion) | ~500 Ko |

---

## 🏆 Remerciements

- **Flipkart** pour le dataset public
- **Hugging Face** pour les modèles transformers
- **Streamlit** pour l'outil de dashboarding
- **Kaggle** pour l'hébergement du dataset

---

> **📅 Projet réalisé en 2025-2026**  
> **Établissement :** [Votre école/université]  
> **Encadrant :** [Nom de l'encadrant]

---

**🎉 Projet complet et prêt pour la soutenance !**