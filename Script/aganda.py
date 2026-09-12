import os
import hashlib
import requests
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION DES DOSSIERS
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed"
)

os.makedirs(DATA_DIR, exist_ok=True)

FILE_PATH = os.path.join(
    DATA_DIR,
    "events.csv"
)


# ============================================================
# CONFIGURATION API
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL")

if not API_BASE_URL:
    print("❌ Erreur : API_BASE_URL n'est pas définie dans .env")
    exit()


DAYS_HISTORY = int(
    os.getenv("DAYS_HISTORY", 365)
)


# ============================================================
# CONSTRUCTION DES PARAMÈTRES API
# ============================================================

def build_api_params(region: str, city: str):
    """
    Construit les paramètres pour l'API OpenDataSoft.
    """

    return {
        # Nombre maximum d'événements récupérés
        "limit": 100,

        # Les événements les plus récents en premier
        "order_by": "firstdate_begin desc",

        # Filtre région + ville
        "refine": [
            f"location_region:{region}",
            f"location_city:{city}"
        ]
    }


# ============================================================
# INPUT UTILISATEUR
# ============================================================

region = input("Région : ").strip()
city = input("Ville : ").strip()


if not region or not city:

    print("--------------------------------")
    print("❌ La région et la ville sont obligatoires.")
    print("--------------------------------")

    exit()


print("\n================================")
print("RECHERCHE D'ÉVÉNEMENTS")
print("================================")

print("Région :", region)
print("Ville :", city)


# ============================================================
# PARAMÈTRES API
# ============================================================

params = build_api_params(
    region,
    city
)


# ============================================================
# RÉCUPÉRATION DES DONNÉES
# ============================================================

try:

    response = requests.get(
        API_BASE_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    json_data = response.json()

    if "results" not in json_data:

        raise ValueError(
            "La réponse API ne contient pas 'results'."
        )

    data = json_data["results"]

    print("\nURL utilisée :")
    print(response.url)

    print(
        "\nNombre d'événements récupérés :",
        len(data)
    )


except requests.exceptions.Timeout:

    print("\n❌ Erreur : délai d'attente dépassé.")

    exit()


except requests.exceptions.RequestException as e:

    print(
        "\n❌ Erreur HTTP/API :",
        e
    )

    exit()


except ValueError as e:

    print(
        "\n❌ Erreur données :",
        e
    )

    exit()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(data)


if df.empty:

    print("--------------------------------")
    print(
        f"Aucun événement trouvé pour "
        f"{city} ({region})."
    )

    print(
        "Aucune donnée ajoutée au fichier events.csv."
    )

    print("--------------------------------")

    exit()


# ============================================================
# VÉRIFICATION DES COLONNES
# ============================================================

required_columns = [
    "title_fr",
    "firstdate_begin",
    "location_city"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print("--------------------------------")

    print(
        "❌ Colonnes manquantes dans la réponse API :"
    )

    for col in missing_columns:
        print("-", col)

    print("--------------------------------")

    exit()


# ============================================================
# NETTOYAGE DES DATES
# ============================================================

df["firstdate_begin"] = pd.to_datetime(
    df["firstdate_begin"],
    errors="coerce",
    utc=True
)


# lastdate_end peut ne pas toujours exister
if "lastdate_end" in df.columns:

    df["lastdate_end"] = pd.to_datetime(
        df["lastdate_end"],
        errors="coerce",
        utc=True
    )

else:

    df["lastdate_end"] = pd.NaT


# Supprimer les événements sans date de début
df = df.dropna(
    subset=["firstdate_begin"]
).copy()


print("\n================================")
print("DIAGNOSTIC DES DATES")
print("================================")

print(
    "Événements après nettoyage :",
    len(df)
)

print(
    "Date la plus ancienne :",
    df["firstdate_begin"].min()
)

print(
    "Date la plus récente :",
    df["firstdate_begin"].max()
)


# ============================================================
# AFFICHER LES 10 ÉVÉNEMENTS LES PLUS RÉCENTS
# ============================================================

print("\n--- 10 ÉVÉNEMENTS LES PLUS RÉCENTS ---")


recent_preview = (
    df[
        [
            "title_fr",
            "firstdate_begin",
            "location_city"
        ]
    ]
    .sort_values(
        "firstdate_begin",
        ascending=False
    )
    .head(10)
)


print(
    recent_preview.to_string(
        index=False
    )
)


# ============================================================
# DATE ACTUELLE UTC
# ============================================================

now = pd.Timestamp.now(
    tz="UTC"
)


# ============================================================
# DATE MINIMALE ACCEPTÉE
# ============================================================

min_date = (
    now -
    pd.Timedelta(
        days=DAYS_HISTORY
    )
)


print("\n--- PÉRIODE DE RECHERCHE ---")

print(
    "Maintenant UTC :",
    now
)

print(
    f"Historique demandé : {DAYS_HISTORY} jours"
)

print(
    "Date minimale :",
    min_date
)


# ============================================================
# FILTRE DATE
# ============================================================

df_new = df[
    df["firstdate_begin"] >= min_date
].copy()


print(
    "\nÉvénements après filtre date :",
    len(df_new)
)


# ============================================================
# SI AUCUN ÉVÉNEMENT
# ============================================================

if df_new.empty:

    print("--------------------------------")

    print(
        f"Aucun événement trouvé entre "
        f"{min_date.strftime('%Y-%m-%d')} "
        f"et les événements à venir pour "
        f"{city} ({region})."
    )

    print(
        "Aucune donnée ajoutée au fichier events.csv."
    )

    print("--------------------------------")

    exit()


# ============================================================
# AFFICHER LES ÉVÉNEMENTS RETENUS
# ============================================================

print("\n================================")
print("ÉVÉNEMENTS RETENUS")
print("================================")


for _, row in (
    df_new
    .sort_values("firstdate_begin")
    .iterrows()
):

    title = row.get(
        "title_fr",
        "Sans titre"
    )

    event_city = row.get(
        "location_city",
        ""
    )

    start_date = row.get(
        "firstdate_begin",
        ""
    )

    print(
        f"\n• {title}"
    )

    print(
        f"  Ville : {event_city}"
    )

    print(
        f"  Date : {start_date}"
    )


# ============================================================
# GÉNÉRATION D'UN UID STABLE
# ============================================================

def generate_uid(row):

    value = (
        str(row.get("title_fr", ""))
        + "|"
        + str(row.get("firstdate_begin", ""))
        + "|"
        + str(row.get("location_city", ""))
    )

    return hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()


df_new["uid"] = df_new.apply(
    generate_uid,
    axis=1
)


# ============================================================
# COLONNES À CONSERVER
# ============================================================

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


# Ajouter uniquement les colonnes présentes
available_columns = [
    col
    for col in columns
    if col in df_new.columns
]


df_clean = df_new[
    available_columns
].copy()


# ============================================================
# NETTOYAGE DES KEYWORDS
# ============================================================

if "keywords_fr" in df_clean.columns:

    df_clean["keywords_fr"] = (
        df_clean["keywords_fr"]
        .apply(
            lambda x:
                ", ".join(map(str, x))
                if isinstance(x, list)
                else (
                    ""
                    if pd.isna(x)
                    else str(x)
                )
        )
    )


# ============================================================
# NETTOYAGE DES COLONNES TEXTUELLES
# ============================================================

for column in [
    "title_fr",
    "description_fr",
    "location_city",
    "category"
]:

    if column in df_clean.columns:

        df_clean[column] = (
            df_clean[column]
            .fillna("")
            .astype(str)
        )


# ============================================================
# TEXTE POUR LES EMBEDDINGS
# ============================================================

df_clean["text_for_embedding"] = (

    "Titre : "
    + df_clean["title_fr"]
    + "\n"

    "Description : "
    + df_clean["description_fr"]
    + "\n"

    "Ville : "
    + df_clean["location_city"]
    + "\n"

    "Début : "
    + df_clean["firstdate_begin"]
        .astype(str)
    + "\n"

    "Fin : "
    + df_clean["lastdate_end"]
        .astype(str)
    + "\n"

    "Catégorie : "
    + df_clean["category"]
    + "\n"

    "Mots-clés : "
    + df_clean["keywords_fr"]
)


# ============================================================
# CHARGER L'HISTORIQUE
# ============================================================

if os.path.exists(FILE_PATH):

    print("\n================================")
    print("FUSION AVEC L'HISTORIQUE")
    print("================================")

    try:

        old_df = pd.read_csv(
            FILE_PATH
        )

        print(
            "Anciens événements :",
            len(old_df)
        )

    except Exception as e:

        print(
            "⚠️ Impossible de lire "
            "l'ancien events.csv :",
            e
        )

        old_df = pd.DataFrame()


else:

    old_df = pd.DataFrame()


# ============================================================
# FUSION
# ============================================================

if not old_df.empty:

    df_final = pd.concat(
        [
            old_df,
            df_clean
        ],
        ignore_index=True
    )

else:

    df_final = df_clean.copy()


# ============================================================
# SUPPRESSION DES DOUBLONS
# ============================================================

if "uid" in df_final.columns:

    before = len(df_final)

    df_final = (
        df_final
        .drop_duplicates(
            subset=["uid"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    after = len(df_final)

    print(
        "Doublons supprimés :",
        before - after
    )


# ============================================================
# SAUVEGARDE CSV
# ============================================================

df_final.to_csv(
    FILE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# RÉSULTAT FINAL
# ============================================================

print("\n================================")
print("SUCCÈS")
print("================================")

print(
    "Nouveaux événements ajoutés :",
    len(df_clean)
)

print(
    "Nombre total d'événements :",
    len(df_final)
)

print(
    "Fichier :",
    FILE_PATH
)

print("================================")
