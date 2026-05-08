# Documentation du Dataset Flipkart

## Source
**Flipkart Product Reviews** — Dataset public de reviews e-commerce provenant de la plateforme **Flipkart (Inde)**.
Disponible sur Kaggle : [Flipkart Product Customer Reviews Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)

## Structure des fichiers

| Fichier | Description |
|---------|-------------|
| `flipkart_data_with_sentiment.csv` | Dataset brut avec labels de sentiment assignés |
| `flipkart_data_preprocessed.csv` | Dataset après preprocessing NLP (tokenisation, stopwords, lemmatisation) |
| `y_train.csv` | Labels d'entraînement (colonne sentiment) |
| `y_test.csv` | Labels de test (colonne sentiment) |
| `X_train.pkl` | Vecteurs TF-IDF d'entraînement (scipy sparse matrix) |
| `X_test.pkl` | Vecteurs TF-IDF de test (scipy sparse matrix) |
| `history.db` | Base SQLite — historique des analyses utilisateur |

## Statistiques du Dataset

| Métrique | Valeur |
|----------|--------|
| Total reviews | **9 976** |
| Train / Test | **7 980 / 1 996** (80% / 20%) |
| Split | `train_test_split(random_state=42, stratify=y)` |

## Distribution des classes

| Classe | Effectif | % | Règle d'assignation |
|--------|----------|---|---------------------|
| **POSITIF** | 8 091 | **81.1%** | rating ≥ 4 étoiles |
| **NÉGATIF** | 1 001 | **10.0%** | rating ≤ 2 étoiles |
| **NEUTRE** | 884 | **8.9%** | rating = 3 étoiles |

> ⚠️ **Dataset déséquilibré** — POSITIF représente 81% des données.
> C'est la principale raison du faible F1-NEUTRE (~0.29 pour LR) : le modèle n'a pas assez d'exemples NEUTRE et NÉGATIF pour bien les apprendre.
>
> **Pistes d'amélioration** : SMOTE, `class_weight="balanced"`, sous-échantillonnage POSITIF, ou augmentation de données.

## Colonnes du dataset brut

| Colonne | Type | Description |
|---------|------|-------------|
| `review` | str | Texte de la review (anglais principalement) |
| `rating` | float | Note de 1 à 5 étoiles |
| `sentiment` | str | Label : Positif / Neutre / Négatif |
| `review_length` | int | Nombre de caractères |
| `lang` | str | Langue détectée (langdetect) |
| `word_count` | int | Nombre de mots |

## Preprocessing appliqué

Pipeline défini dans `src/preprocessing.py` :

1. **`lower()`** — mise en minuscules
2. **`re.sub(r'[^a-z\s]', '', text)`** — suppression ponctuation et chiffres
3. **`word_tokenize`** — tokenisation NLTK
4. **Suppression des stopwords anglais** (NLTK)
5. **`WordNetLemmatizer`** — lemmatisation
6. **`TfidfVectorizer(max_features=10000, ngram_range=(1,2))`** — vectorisation

## Reproductibilité

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Le `random_state=42` garantit des splits identiques à chaque exécution.

## Schéma de la base SQLite (`history.db`)

```sql
CREATE TABLE logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,
    texte_ou_url  TEXT NOT NULL,
    modele_utilise TEXT NOT NULL,
    v2_model_key  TEXT,
    sentiment     TEXT,
    confiance     REAL,
    source        TEXT,
    type_analyse  TEXT NOT NULL
);
```

Cette table stocke l'historique de toutes les analyses (prédictions individuelles et scrapings) effectuées via le dashboard.