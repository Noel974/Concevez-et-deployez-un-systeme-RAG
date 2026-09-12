import sys
import os

# Ajouter le dossier racine au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pandas as pd
from utils.mistral import embed_texts

# Chemins basés sur ROOT_DIR : le script fonctionne quel que soit le
# dossier depuis lequel il est lancé (avant : chemins relatifs fragiles).
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
EVENTS_FILE = os.path.join(DATA_DIR, "events.csv")
VECTORS_FILE = os.path.join(DATA_DIR, "events_vectors.pkl")


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
    "Sauvegardé dans :",
    VECTORS_FILE
)
print("--------------------------------")