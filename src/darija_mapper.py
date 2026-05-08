# src/darija_mapper.py
# ═══════════════════════════════════════════════════════════════════════════════
# Mapping Darija (romanisée) → Français pour améliorer l'analyse de sentiment
# ═══════════════════════════════════════════════════════════════════════════════

import re

# Dictionnaire principal : Darija → Français
DARIJA_TO_FR = {
    # ── Sentiments positifs ───────────────────────────────────────────────────
    "mzyan": "bon", "mzyana": "bonne", "mzyanine": "bonnes",
    "wa3er": "excellent", "wa3ra": "excellente",
    "zwin": "beau", "zwina": "belle",
    "mezyan": "bien", "mezyana": "bien",
    "safi": "ça va", "kulshi mezyan": "tout va bien",
    "hamdullah": "heureusement", "lhmd": "heureusement",
    "bzzaf": "beaucoup", "bzf": "beaucoup",
    "waaaa": "très", "waaaw": "super",
    "3ziz": "cher", "3ziza": "chère",
    "nadi": "propre", "nadia": "propre",
    "3jbni": "j'aime", "3jebni": "j'aime",
    "tbarkallah": "magnifique", "tbark lah": "magnifique",
    "merci": "merci", "chokran": "merci", "choukran": "merci",
    "bravo": "bravo", "yalah": "allez",
    
    # ── Sentiments négatifs ───────────────────────────────────────────────────
    "khayb": "mauvais", "khayba": "mauvaise", "khaybin": "mauvais",
    "7ram": "honteux", "hram": "honteux",
    "zamel": "nul", "zamela": "nulle",
    "mabghitch": "je ne veux pas", "ma bghitch": "je ne veux pas",
    "makhdamch": "ne fonctionne pas", "ma khdamch": "ne fonctionne pas",
    "mafhemch": "je ne comprends pas", "ma fhemch": "je ne comprends pas",
    "ma3endich": "je n'ai pas", "ma 3endich": "je n'ai pas",
    "mamsalitch": "ne ressemble pas", "ma msalitch": "ne ressemble pas",
    "3yan": "fatigué", "3yana": "fatiguée", "3yanin": "fatigués",
    "sda3": "trop bruyant", "sdaa": "trop bruyante",
    "t7e9er": "mépris", "t7e9ri": "mépris",
    "machi": "pas", "machi mzyan": "pas bon",
    "ma3ndich": "je n'ai pas", "ma3ndich flous": "pas d'argent",
    "flous": "argent", "floos": "argent", "floss": "argent",
    "ghali": "cher", "ghalia": "chère", "ghalin": "chers",
    "ri9": "vide", "r9i9": "mince", "r9a9": "mince",
    "m3e9er": "difficile", "m3e9ra": "difficile",
    "mab9atch": "n'existe plus", "ma b9atch": "n'existe plus",
    "mamsafitch": "ne marche pas", "ma msafitch": "ne marche pas",
    "mazal": "encore", "mazal khayb": "toujours mauvais",
    "mochkil": "problème", "mochkile": "problème", "mochkilat": "problèmes",
    "3ya9": "ennui", "3ya9a": "ennuyeuse",
    "khra": "mauvais", "khraa": "mauvaise",  # attention contextuel
    "ti7": "tombe", "tay7": "tombe",
    "t9e3": "casse", "t9e3a": "cassée",
    "maksour": "cassé", "maksoura": "cassée",
    "majmoud": "moisi", "majmouda": "moisie",
    
    # ── Verbes d'action ───────────────────────────────────────────────────────
    "srite": "j'ai acheté", "chrite": "j'ai acheté", "chrit": "j'ai acheté",
    "srit": "j'ai acheté", "chriti": "tu as acheté", "sriti": "tu as acheté",
    "wasal": "arrivé", "wasalni": "arrivé à moi", "wesel": "arrivé",
    "jani": "il m'est arrivé", "jani l": "il m'est arrivé le",
    "3tani": "il m'a donné", "3tani l": "il m'a donné le",
    "dert": "j'ai fait", "derti": "tu as fait", "dert l": "j'ai fait le",
    "9rit": "j'ai lu", "9riti": "tu as lu",
    "7tit": "j'ai mis", "7titi": "tu as mis",
    "jbt": "j'ai apporté", "jebet": "j'ai apporté",
    "3tini": "donne-moi", "3tini l": "donne-moi le",
    "sift": "envoyer", "sifti": "envoyer", "sift l": "envoyer le",
    "3aweni": "aide-moi", "3aweni l": "aide-moi le",
    "3ref": "savoir", "3refet": "je savais", "ma 3refet": "je ne savais pas",
    "fhemet": "j'ai compris", "ma fhemet": "je n'ai pas compris",
    
    # ── Pronoms & connecteurs ─────────────────────────────────────────────────
    "men": "de", "men l": "du", "men l'": "du",
    "w": "et", "ou": "et",
    "f": "dans", "fe": "dans", "f l": "dans le",
    "b": "avec", "be": "avec", "b l": "avec le",
    "l": "le", "l'": "le",
    "d": "de", "d l": "du",
    "chi": "quelque", "chi wahd": "quelqu'un", "chi haja": "quelque chose",
    "wahd": "un", "wahda": "une", "whed": "un",
    "hada": "ceci", "hadi": "celle-ci", "hadok": "ceux-là",
    "hada l": "ce", "hadi l": "cette",
    "kifach": "comment", "kifesh": "comment", "kifach l": "comment le",
    "3lash": "pourquoi", "3lash l": "pourquoi le",
    "fin": "où", "fenn": "où", "fenn l": "où le",
    "wakha": "même si", "wakha l": "même si le",
    "ila": "si", "ila l": "si le",
    "b7al": "comme", "b7al l": "comme le", "b7al chi": "comme quelque",
    "7it": "parce que", "7it l": "parce que le",
    "walo": "rien", "walo l": "rien de",
    "kolo": "tout", "kolo l": "tout le",
    "b3d": "après", "b3d l": "après le",
    "9bel": "avant", "9bel l": "avant le",
    "daba": "maintenant", "daba l": "maintenant le",
    "dimna": "toujours", "dimna l": "toujours le",
    "yalla": "allez", "yallah": "allez",
    
    # ── Nombres & quantités ──────────────────────────────────────────────────
    "jouj": "deux", "telt": "trois", "reb3": "quatre", "khamsa": "cinq",
    "jouj nhar": "deux jours", "f 2": "en 2", "f 3": "en 3",
    
    # ── Expressions courantes e-commerce ─────────────────────────────────────
    "tawila": "longue", "tawil": "long",
    "9asira": "courte", "9asir": "court",
    "kbira": "grande", "kbir": "grand",
    "sgira": "petite", "sgir": "petit",
    "9dima": "ancienne", "9dim": "ancien",
    "jdida": "nouvelle", "jdid": "nouveau",
    "n9iya": "propre", "n9i": "propre",
    "wesekh": "sale", "weskha": "sale",
    "m3ell9": "suspendu", "m3ell9a": "suspendue",
    "mde9": "collé", "mde9a": "collée",
    "mferre9": "séparé", "mferre9a": "séparée",
    "m7ebbes": "bloqué", "m7ebbesa": "bloquée",
    "mferme3": "fermé", "mferme3a": "fermée",
    "m7loul": "ouvert", "m7loula": "ouverte",
    "mchadda": "serrée", "mchadd": "serré",
    "mrte7": "relâché", "mrte7a": "relâchée",
    "m3ebber": "rempli", "m3ebbera": "remplie",
    "mferregh": "vide", "mferregha": "vide",
}


def translate_darija(text: str) -> str:
    """
    Traduit les mots Darija romanisés en français.
    Gère la casse et la ponctuation attachée.
    """
    if not text:
        return text
    
    words = text.split()
    translated = []
    
    for word in words:
        # Nettoyer la ponctuation pour la recherche
        clean = word.lower().strip(".,!?;:\"'()[]{}")
        
        # Chercher dans le dictionnaire
        if clean in DARIJA_TO_FR:
            # Préserver la casse approximative
            replacement = DARIJA_TO_FR[clean]
            # Remettre la ponctuation si elle existait
            if word[-1] in ".,!?;:\"'()[]{}":
                replacement += word[-1]
            translated.append(replacement)
        else:
            translated.append(word)
    
    return " ".join(translated)


def detect_darija(text: str) -> bool:
    """
    Détecte si un texte contient des mots Darija.
    Retourne True si >20% des mots reconnus sont Darija.
    """
    if not text:
        return False
    
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return False
    
    darija_count = sum(1 for w in words if w in DARIJA_TO_FR)
    return darija_count / len(words) > 0.15  # 15% de mots Darija = seuil


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RAPIDE (si exécuté directement)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        "Mzyan bzzaf had l'produit, srite men Jumia w wasalni f 2 jours !",
        "Khayb had l'produit, mafhemch chno dert b had l flous, 7ram.",
        "Wa3er l'produit mais l prix bzzaf, ma3endich flous bzzaf.",
        "Ce produit est excellent, je recommande !",  # Pas de Darija
    ]
    
    print("═" * 70)
    print("Test du mapper Darija")
    print("═" * 70)
    for t in tests:
        is_darija = detect_darija(t)
        translated = translate_darija(t)
        print(f"\nOriginal    : {t}")
        print(f"Darija?     : {'OUI' if is_darija else 'NON'}")
        print(f"Traduit     : {translated}")
    print("═" * 70)