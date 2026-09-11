"""
Script de vectorisation des événements pour le pipeline RAG.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
from utils.mistral import embed_texts


# ---------------------------------------------------------
# Chemins
# ---------------------------------------------------------

EVENTS_FILE = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "events.csv"
)

VECTORS_FILE = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "events_vectors.pkl"
)


# ---------------------------------------------------------
# Chargement des événements
# ---------------------------------------------------------

df = pd.read_csv(EVENTS_FILE)

print("Nombre total d'événements :", len(df))


# Vérification du fichier events.csv
if "uid" not in df.columns:
    raise ValueError(
        "Erreur : la colonne 'uid' est absente de events.csv."
    )

if "text_for_embedding" not in df.columns:
    raise ValueError(
        "Erreur : la colonne 'text_for_embedding' est absente de events.csv."
    )


# ---------------------------------------------------------
# Chargement des anciennes vectorisations
# ---------------------------------------------------------

old_vectors = pd.DataFrame()

if os.path.exists(VECTORS_FILE):

    try:
        old_vectors = pd.read_pickle(VECTORS_FILE)

        print(
            "Anciennes vectorisations trouvées :",
            len(old_vectors)
        )

        # Vérifier que le pickle contient bien les colonnes nécessaires
        if (
            old_vectors.empty
            or "uid" not in old_vectors.columns
            or "vector" not in old_vectors.columns
        ):
            print(
                "Ancien fichier de vecteurs invalide ou vide."
            )

            # On repart de zéro
            old_vectors = pd.DataFrame()

    except Exception as e:

        print(
            "Impossible de lire l'ancien fichier de vecteurs :",
            e
        )

        old_vectors = pd.DataFrame()


# ---------------------------------------------------------
# Identifier les nouveaux événements
# ---------------------------------------------------------

if old_vectors.empty:

    new_events = df.copy()

else:

    new_events = df[
        ~df["uid"].isin(old_vectors["uid"])
    ].copy()


print(
    "Nouveaux événements à vectoriser :",
    len(new_events)
)


# ---------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------

if len(new_events) > 0:

    vectors = embed_texts(
        new_events["text_for_embedding"].tolist()
    )

    new_events["vector"] = vectors


# ---------------------------------------------------------
# Fusion avec les anciennes vectorisations
# ---------------------------------------------------------

if old_vectors.empty:

    df_vectors = new_events

elif new_events.empty:

    df_vectors = old_vectors

else:

    df_vectors = pd.concat(
        [old_vectors, new_events],
        ignore_index=True
    )


# ---------------------------------------------------------
# Suppression des doublons
# ---------------------------------------------------------

if not df_vectors.empty:

    df_vectors = df_vectors.drop_duplicates(
        subset=["uid"],
        keep="last"
    )


# ---------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------

df_vectors.to_pickle(VECTORS_FILE)


print("--------------------------------")
print(
    "Nombre total de vecteurs :",
    len(df_vectors)
)
print(
    "Sauvegardé dans :",
    VECTORS_FILE
)
print("--------------------------------")
