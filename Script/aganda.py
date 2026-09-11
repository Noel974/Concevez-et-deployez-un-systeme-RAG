"""
Module de pré-processing des événements OpenAgenda.

Ce script :
- construit dynamiquement l’URL API selon la région et la ville,
- récupère les événements via l’API OpenDataSoft,
- nettoie et convertit les dates,
- filtre les événements selon une période définie,
- sélectionne les colonnes utiles,
- génère un texte optimisé pour la vectorisation,
- fusionne les nouvelles données avec l’historique existant,
- sauvegarde le résultat dans data/processed/events.csv.

Ce module constitue la première étape du pipeline RAG.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
DAYS_HISTORY = int(os.getenv("DAYS_HISTORY", 365))

FILE_PATH = "data/processed/events.csv"


def build_api_url(region: str, city: str) -> str:
    """
    Construit l'URL de requête OpenDataSoft pour filtrer les événements
    par région et ville.

    Args:
        region (str): Région à filtrer.
        city (str): Ville à filtrer.

    Returns:
        str: URL complète prête à être appelée via requests.
    """
    return (
        f"{API_BASE_URL}"
        f"?refine=location_region:{region}"
        f"&refine=location_city:{city}"
    )


# --- Choix utilisateur ---
region = input("Région : ")
city = input("Ville : ")

API_URL = build_api_url(region, city)
print("URL utilisée :", API_URL)


# --- Récupération des données ---
try:
    response = requests.get(API_URL, timeout=20)
    response.raise_for_status()

    json_data = response.json()

    if "results" not in json_data:
        raise ValueError("La réponse API ne contient pas 'results'.")

    data = json_data["results"]

except Exception as e:
    print("Erreur récupération API :", e)
    exit()


# --- Création DataFrame ---
df = pd.DataFrame(data)

if df.empty:
    print("Aucun événement trouvé.")
    exit()

print("Nombre d'événements récupérés :", len(df))


# --- Nettoyage des dates ---
df["firstdate_begin"] = (
    pd.to_datetime(
        df["firstdate_begin"],
        errors="coerce",
        utc=True
    )
    .dt.tz_convert(None)
)

df["lastdate_end"] = (
    pd.to_datetime(
        df["lastdate_end"],
        errors="coerce",
        utc=True
    )
    .dt.tz_convert(None)
)


# --- Filtre période ---
min_date = datetime.now() - timedelta(days=DAYS_HISTORY)
df = df[df["lastdate_end"] >= min_date]


# --- Sélection des colonnes utiles ---
columns = [
    "uid",
    "title_fr",
    "description_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_city",
    "category",
    "keywords_fr"
]

available_columns = [c for c in columns if c in df.columns]
df_clean = df[available_columns].copy()


# --- Création du texte pour embeddings ---
df_clean["text_for_embedding"] = (
    df_clean["title_fr"].fillna("")
    + "\n"
    + df_clean["description_fr"].fillna("")
    + "\nVille : "
    + df_clean["location_city"].fillna("")
    + "\nDate : "
    + df_clean["firstdate_begin"].astype(str)
)


# ==================================================
# AJOUT AUX DONNÉES EXISTANTES
# ==================================================

os.makedirs("data/processed", exist_ok=True)

if os.path.exists(FILE_PATH):

    print("Ancien fichier trouvé, fusion en cours...")

    old_df = pd.read_csv(FILE_PATH)

    df_final = pd.concat([old_df, df_clean], ignore_index=True)

    # Suppression des doublons
    if "uid" in df_final.columns:
        df_final = df_final.drop_duplicates(subset=["uid"], keep="last")

else:

    print("Aucun historique trouvé, création du fichier.")
    df_final = df_clean


# --- Sauvegarde ---
df_final.to_csv(FILE_PATH, index=False)

print("--------------------------------")
print("Fichier mis à jour :", FILE_PATH)
print("Nombre total événements :", len(df_final))
print("--------------------------------")

print(df_final.head())
