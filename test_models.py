"""
test_models.py — Test direct des modèles (sans API)
════════════════════════════════════════════════════
Usage : python test_models.py
════════════════════════════════════════════════════
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import TransformerPredictor

# Charger les 3 modèles
print("⏳ Chargement des modèles...")
models = {}
for key in ["default", "twitter", "french"]:
    try:
        models[key] = TransformerPredictor(model_key=key)
        print(f"  ✅ {key} chargé")
    except Exception as e:
        print(f"  ❌ {key} échoué : {e}")

# Phrases de test
tests = [
    ("FR positif", "Ce produit est excellent, je recommande !"),
    ("FR négatif", "Produit de mauvaise qualité, très déçu."),
    ("Darija pos", "Mzyan bzzaf, wa3er l'produit"),
    ("Darija neg", "Khayb w 7ram, mafhemch chno hada"),
    ("Sarcasme", "Oh génial, encore un truc qui marche pas."),
    ("Emoji", "😍😍 trop bien j'adore"),
    ("Neutre", "Le colis est arrivé le mardi."),
]

print("\n" + "═" * 80)
print(f"{'Test':<20} {'default':<20} {'twitter':<20} {'french':<20}")
print("═" * 80)

for name, text in tests:
    row = f"{name:<20}"
    for key in ["default", "twitter", "french"]:
        if key in models:
            r = models[key].predict(text)
            row += f"{r['sentiment']:<8}({r['confidence']:.0%})   "
        else:
            row += f"{'N/A':<20}"
    print(row)

print("═" * 80)