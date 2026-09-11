import sys
import os

# Ajouter le dossier racine au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


import pandas as pd
from utils.mistral import embed_texts


EVENTS_FILE = "data/processed/events.csv"
VECTORS_FILE = "data/processed/events_vectors.pkl"


# --- Charger les événements ---
df = pd.read_csv(EVENTS_FILE)

print("Nombre total d'événements :", len(df))


# ==================================================
# Charger les anciens embeddings s'ils existent
# ==================================================

if os.path.exists(VECTORS_FILE):

    old_vectors = pd.read_pickle(VECTORS_FILE)

    print(
        "Anciennes vectorisations trouvées :",
        len(old_vectors)
    )


    # Trouver les nouveaux événements
    new_events = df[
        ~df["uid"].isin(old_vectors["uid"])
    ]


else:

    print("Aucun embedding existant.")

    old_vectors = pd.DataFrame()

    new_events = df



# ==================================================
# Vectoriser uniquement les nouveaux événements
# ==================================================

if len(new_events) > 0:

    print(
        "Nouveaux événements à vectoriser :",
        len(new_events)
    )


    vectors = embed_texts(
        new_events["text_for_embedding"].tolist()
    )


    new_events = new_events.copy()

    new_events["vector"] = vectors


else:

    print("Aucun nouvel événement.")

    new_events = pd.DataFrame()



# ==================================================
# Fusion ancien + nouveau
# ==================================================

if not old_vectors.empty:

    df_vectors = pd.concat(
        [
            old_vectors,
            new_events
        ],
        ignore_index=True
    )

else:

    df_vectors = new_events



# Supprimer les doublons éventuels
df_vectors = df_vectors.drop_duplicates(
    subset=["uid"],
    keep="last"
)



# --- Sauvegarde ---
df_vectors.to_pickle(
    VECTORS_FILE
)


print("--------------------------------")
print(
    "Nombre total de vecteurs :",
    len(df_vectors)
)

print(
    "Sauvegardé dans :","""
Script de vectorisation des événements pour le pipeline RAG.

Ce module :
- charge les événements pré‑processés depuis events.csv,
- détecte les événements déjà vectorisés (via events_vectors.pkl),
- vectorise uniquement les nouveaux événements grâce à la fonction embed_texts(),
- fusionne les anciens et nouveaux vecteurs,
- supprime les doublons,
- sauvegarde le fichier final events_vectors.pkl.

Ce script constitue l'étape 2 du pipeline RAG :
1. ingestion & pré-processing (agenda.py)
2. vectorisation (embed.py)
3. indexation FAISS (index.py)
"""

import sys
import os

# Ajouter le dossier racine au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
from utils.mistral import embed_texts


EVENTS_FILE = "data/processed/events.csv"
VECTORS_FILE = "data/processed/events_vectors.pkl"


# --- Charger les événements ---
df = pd.read_csv(EVENTS_FILE)
print("Nombre total d'événements :", len(df))


# ==================================================
# Charger les anciens embeddings s'ils existent
# ==================================================

if os.path.exists(VECTORS_FILE):

    old_vectors = pd.read_pickle(VECTORS_FILE)
    print("Anciennes vectorisations trouvées :", len(old_vectors))

    # Trouver les nouveaux événements
    new_events = df[~df["uid"].isin(old_vectors["uid"])]

else:

    print("Aucun embedding existant.")
    old_vectors = pd.DataFrame()
    new_events = df


# ==================================================
# Vectoriser uniquement les nouveaux événements
# ==================================================

if len(new_events) > 0:

    print("Nouveaux événements à vectoriser :", len(new_events))

    vectors = embed_texts(new_events["text_for_embedding"].tolist())

    new_events = new_events.copy()
    new_events["vector"] = vectors

else:

    print("Aucun nouvel événement.")
    new_events = pd.DataFrame()


# ==================================================
# Fusion ancien + nouveau
# ==================================================

if not old_vectors.empty:
    df_vectors = pd.concat([old_vectors, new_events], ignore_index=True)
else:
    df_vectors = new_events

# Supprimer les doublons éventuels
df_vectors = df_vectors.drop_duplicates(subset=["uid"], keep="last")


# --- Sauvegarde ---
df_vectors.to_pickle(VECTORS_FILE)

print("--------------------------------")
print("Nombre total de vecteurs :", len(df_vectors))
print("Sauvegardé dans :", VECTORS_FILE)
print("--------------------------------")

    VECTORS_FILE
)
print("--------------------------------")