
# src/evaluate.py — métriques partagées
# Auteurs : Aymen Ichqarrane & Ihssane Moutchou

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def rapport_complet(nom: str, y_test, y_pred) -> dict:
    """
    Calcule et affiche toutes les métriques pour un modèle.
    Retourne un dict de résultats pour la comparaison finale.
    """
    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'='*50}")
    print(f"  {nom}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 weighted : {f1_w:.4f}")
    print(f"  F1 macro    : {f1_m:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {
        'nom'         : nom,
        'accuracy'    : acc,
        'f1_weighted' : f1_w,
        'f1_macro'    : f1_m,
    }

def tableau_comparatif(liste_resultats: list) -> pd.DataFrame:
    """
    Prend une liste de dicts retournés par rapport_complet()
    et retourne un DataFrame trié par F1 weighted décroissant.
    """
    df = pd.DataFrame(liste_resultats)
    df = df.sort_values('f1_weighted', ascending=False).reset_index(drop=True)
    df.index += 1  # rang commence à 1
    return df