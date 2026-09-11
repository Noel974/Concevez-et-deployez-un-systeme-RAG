"""
Module de pré-processing des événements OpenAgenda.

Ce script :
- construit l’URL API selon la région et la ville,
- récupère les événements,
- nettoie les dates,
- filtre les événements récents,
- génère un texte pour embeddings,
- fusionne avec l’historique,
- sauvegarde events.csv dans data/processed/.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Déterminer le dossier racine du projet
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

FILE_PATH = os.path.join(DATA_DIR, "events.csv")

API_BASE_URL = os.getenv("API_BASE_URL")
DAYS_HISTORY = int(os.getenv("DAYS_HISTORY", 365))


def build_api_url(region: str, city: str) -> str:
    """Construit l'URL API OpenDataSoft."""
    return (
        f"{API_BASE_URL}"
        f"?refine=location_region:{region}"
        f"&refine=location_city:{city}"
    )


# --- Input utilisateur ---
region = input("Région : ")
city = input("Ville : ")

API_URL = build_api_url(region, city)
print("URL utilisée :", API_URL)


# --- Récupération API ---
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


# --- DataFrame ---
df = pd.DataFrame(data)
if df.empty:
    print("Aucun événement trouvé.")
    exit()

print("Nombre d'événements récupérés :", len(df))


# --- Nettoyage dates ---
df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], errors="coerce", utc=True).dt.tz_convert(None)
df["lastdate_end"] = pd.to_datetime(df["lastdate_end"], errors="coerce", utc=True).dt.tz_convert(None)


# --- Filtre période ---
min_date = datetime.now() - timedelta(days=DAYS_HISTORY)
df = df[df["firstdate_begin"] >= min_date]


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

df_clean = df[[c for c in columns if c in df.columns]].copy()

# Générer un UID si absent
if "uid" not in df_clean.columns:
    df_clean["uid"] = df_clean.apply(
        lambda row: hash(
            str(row.get("title_fr", "")) +
            str(row.get("firstdate_begin", "")) +
            str(row.get("location_city", ""))
        ),
        axis=1
    )


# --- Texte embeddings ---
df_clean["text_for_embedding"] = (
    df_clean["title_fr"].fillna("") + "\n" +
    df_clean["description_fr"].fillna("") + "\nVille : " +
    df_clean["location_city"].fillna("") + "\nDate : " +
    df_clean["firstdate_begin"].astype(str)
)


# --- Fusion historique ---
if os.path.exists(FILE_PATH):
    old_df = pd.read_csv(FILE_PATH)
    df_final = pd.concat([old_df, df_clean], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["uid"], keep="last")
else:
    df_final = df_clean


# --- Sauvegarde ---
df_final.to_csv(FILE_PATH, index=False)

print("--------------------------------")
print("Fichier mis à jour :", FILE_PATH)
print("Nombre total événements :", len(df_final))
print("--------------------------------")
