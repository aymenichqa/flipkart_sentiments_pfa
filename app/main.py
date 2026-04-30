from src.models import charger_vectorizer, charger_champion, predire
from src.preprocessing import clean_text

vectorizer = charger_vectorizer()
modele     = charger_champion()

@app.post("/predict")
def predict(request: PredictionRequest):
    texte_propre = clean_text(request.text)      # même nettoyage qu'à l'entraînement
    X = vectorizer.transform([texte_propre])
    label, probas = predire(X, modele)
    return {"sentiment": label, "confidence": max(probas.values()), "probabilities": probas}