data/ — Documentation du dataset
Source
Flipkart Product Reviews — dataset public de reviews e-commerce provenant
de la plateforme Flipkart (Inde). Disponible sur Kaggle :
https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset
Format
FichierDescriptionflipkart_data_with_sentiment.csvDataset brut avec labels de sentiment assignésflipkart_data_preprocessed.csvDataset après preprocessing NLP (tokenisation, stopwords, lemmatisation)y_train.csvLabels d'entraînement (colonne sentiment)y_test.csvLabels de test (colonne sentiment)X_train.pklVecteurs TF-IDF d'entraînement (scipy sparse matrix)X_test.pklVecteurs TF-IDF de test (scipy sparse matrix)
Statistiques
MétriqueValeurTotal reviews9 976Train / Test7 980 / 1 996 (80% / 20%)Splittrain_test_split(random_state=42, stratify=y)
Distribution des classes
ClasseEffectif%Règle d'assignationPOSITIF8 09181.1%rating ≥ 4 étoilesNÉGATIF1 00110.0%rating ≤ 2 étoilesNEUTRE8848.9%rating = 3 étoiles

⚠️ Dataset déséquilibré — POSITIF représente 81% des données.
C'est la principale raison du faible F1-NEUTRE (~0.29 pour LR) : le modèle
n'a pas assez d'exemples NEUTRE et NÉGATIF pour bien les apprendre.
Pistes d'amélioration : SMOTE, class_weight="balanced", sous-échantillonnage POSITIF.

Colonnes du dataset brut
ColonneTypeDescriptionreviewstrTexte de la review en anglaisratingfloatNote de 1 à 5 étoilessentimentstrLabel : Positif / Neutre / Négatifreview_lengthintNombre de caractèreslangstrLangue détectée (langdetect)word_countintNombre de mots
Preprocessing appliqué (src/preprocessing.py)

lower() — mise en minuscules
re.sub(r'[^a-z\s]', '', text) — suppression ponctuation et chiffres
word_tokenize — tokenisation NLTK
Suppression des stopwords anglais (NLTK)
WordNetLemmatizer — lemmatisation
TfidfVectorizer(max_features=10000, ngram_range=(1,2)) — vectorisation

Reproductibilité
pythonfrom sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Le random_state=42 garantit des splits identiques à chaque exécution.
