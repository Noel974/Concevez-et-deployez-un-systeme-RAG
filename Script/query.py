import os
import pandas as pd
import numpy as np
import faiss

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

FAISS_PATH = "data/processed/faiss_index.bin"
DATA_PATH = "data/processed/events_with_vectors.pkl"


# ============================================================
# CHARGEMENT DE FAISS
# ============================================================

if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(
        f"Index FAISS introuvable : {FAISS_PATH}"
    )

index = faiss.read_index(FAISS_PATH)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Fichier de métadonnées introuvable : {DATA_PATH}"
    )

df = pd.read_pickle(DATA_PATH)

print(
    f"Nombre d'événements chargés : {len(df)}"
)


# ============================================================
# VÉRIFICATION DES DONNÉES
# ============================================================

required_columns = [
    "title_fr",
    "description_fr",
    "location_city",
    "firstdate_begin"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(missing_columns)
    )


# ============================================================
# NETTOYAGE DES VILLES
# ============================================================

df["location_city"] = (
    df["location_city"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# EMBEDDINGS MISTRAL
# ============================================================

embedder = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv(
        "MISTRAL_API_KEY"
    )
)


# ============================================================
# MODÈLE MISTRAL
# ============================================================

llm = ChatMistralAI(
    model="open-mistral-7b",
    mistral_api_key=os.getenv(
        "MISTRAL_API_KEY"
    )
)


# ============================================================
# LISTE DES VILLES
# ============================================================

CITIES = (
    df["location_city"]
    .replace("", pd.NA)
    .dropna()
    .unique()
    .tolist()
)

# Les villes les plus longues d'abord
# Exemple : "Saint-Pierre" avant "Pierre"
CITIES = sorted(
    CITIES,
    key=len,
    reverse=True
)


# ============================================================
# DÉTECTION DE LA VILLE
# ============================================================

def detect_city(query):
    """
    Détecte une ville présente dans la question.

    Exemple :
        "Quel événement sur Lyon ?"
        -> Lyon

        "Quels événements à Saint-Pierre ?"
        -> Saint-Pierre
    """

    query_lower = (
        query
        .lower()
        .strip()
    )

    for city in CITIES:

        city_lower = city.lower()

        if city_lower in query_lower:

            return city

    return None


# ============================================================
# RECHERCHE FAISS
# ============================================================

def search_events(query, k=10):
    """
    Recherche sémantique avec FAISS.
    """

    q_vec = np.array(
        embedder.embed_query(query),
        dtype="float32"
    ).reshape(1, -1)

    distances, indices = index.search(
        q_vec,
        k
    )

    valid_indices = [
        i
        for i in indices[0]
        if 0 <= i < len(df)
    ]

    if not valid_indices:
        return pd.DataFrame()

    results = df.iloc[
        valid_indices
    ].copy()

    return results


# ============================================================
# RECHERCHE INTELLIGENTE
# ============================================================

def search_events_smart(
    query,
    k=5
):
    """
    Recherche hybride :

    1. Détection de la ville.
    2. Recherche FAISS.
    3. Filtrage strict sur la ville.
    4. Tri par date.
    5. Retour des k premiers résultats.
    """

    city = detect_city(query)

    print(
        f"\n🔎 Ville détectée : {city}"
    )

    # --------------------------------------------------------
    # CAS 1 : aucune ville détectée
    # --------------------------------------------------------

    if city is None:

        results = search_events(
            query,
            k=k
        )

        return results


    # --------------------------------------------------------
    # CAS 2 : ville détectée
    # --------------------------------------------------------

    # On récupère beaucoup plus de résultats
    # afin d'avoir de meilleures chances de trouver
    # tous les événements de la ville.
    faiss_results = search_events(
        query,
        k=100
    )


    # --------------------------------------------------------
    # FILTRE STRICT SUR LA VILLE
    # --------------------------------------------------------

    city_results = faiss_results[
        faiss_results["location_city"]
        .str.lower()
        == city.lower()
    ].copy()


    # --------------------------------------------------------
    # SI DES ÉVÉNEMENTS SONT TROUVÉS
    # --------------------------------------------------------

    if not city_results.empty:

        print(
            f"✅ {len(city_results)} événement(s) "
            f"trouvé(s) à {city}"
        )

        # Conversion des dates
        city_results["firstdate_begin"] = (
            pd.to_datetime(
                city_results["firstdate_begin"],
                errors="coerce",
                utc=True
            )
        )

        # Trier par date
        city_results = (
            city_results
            .sort_values(
                "firstdate_begin"
            )
        )

        return city_results.head(k)


    # --------------------------------------------------------
    # CAS 3 : FAISS n'a pas trouvé la ville
    # --------------------------------------------------------

    print(
        f"⚠️ Aucun événement de {city} "
        f"dans les résultats FAISS."
    )


    # Recherche directe dans les métadonnées
    direct_results = df[
        df["location_city"]
        .str.lower()
        == city.lower()
    ].copy()


    if not direct_results.empty:

        print(
            f"✅ {len(direct_results)} événement(s) "
            f"trouvé(s) directement dans la base "
            f"pour {city}"
        )

        direct_results[
            "firstdate_begin"
        ] = pd.to_datetime(
            direct_results["firstdate_begin"],
            errors="coerce",
            utc=True
        )

        direct_results = (
            direct_results
            .sort_values(
                "firstdate_begin"
            )
        )

        return direct_results.head(k)


    # --------------------------------------------------------
    # CAS 4 : aucun événement
    # --------------------------------------------------------

    print(
        f"❌ Aucun événement trouvé à {city}"
    )

    return pd.DataFrame()


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def build_context(results):

    if results.empty:

        return (
            "Aucun événement correspondant "
            "n'a été trouvé dans la base."
        )

    context = ""

    for i, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        context += f"""

### ÉVÉNEMENT {i}

Titre :
{row.get("title_fr", "")}

Ville :
{row.get("location_city", "")}

Catégorie :
{row.get("category", "")}

Date de début :
{row.get("firstdate_begin", "")}

Date de fin :
{row.get("lastdate_end", "")}

Description :
{row.get("description_fr", "")}

Mots-clés :
{row.get("keywords_fr", "")}

"""


    return context


# ============================================================
# CHATBOT
# ============================================================

def ask_chatbot(question):

    # --------------------------------------------------------
    # Recherche
    # --------------------------------------------------------

    results = search_events_smart(
        question,
        k=5
    )


    # --------------------------------------------------------
    # Contexte
    # --------------------------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------------------------
    # Ville
    # --------------------------------------------------------

    city = detect_city(
        question
    )


    if city:

        city_instruction = f"""

La question concerne explicitement
la ville de : {city}

RÈGLE ABSOLUE :

Tous les événements proposés doivent être
situés à {city}.

Ne propose aucun événement situé dans
une autre ville.

"""

    else:

        city_instruction = ""


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""

Tu es un assistant spécialisé dans les événements.

Ta mission est de répondre à la question de
l'utilisateur uniquement à partir des événements
présents dans le contexte.

{city_instruction}

RÈGLES :

1. Utilise uniquement les informations du contexte.

2. N'invente jamais d'événement.

3. N'invente jamais une date, une ville ou une description.

4. Si une ville est mentionnée dans la question,
   respecte strictement cette ville.

5. Si plusieurs événements correspondent,
   présente-les sous forme de liste.

6. Si aucun événement correspondant n'est présent,
   dis simplement :
   "Je n'ai trouvé aucun événement correspondant
   dans la base."

7. Ne dis jamais qu'une ville ne possède aucun
   événement si un événement de cette ville
   apparaît dans le contexte.

8. Réponds en français.

9. Pour chaque événement, indique :
   - le titre
   - la ville
   - la date
   - une courte description

10. Sois clair et concis.

============================================================
CONTEXTE
============================================================

{context}

============================================================
QUESTION
============================================================

{question}

============================================================
RÉPONSE
============================================================

"""


    # --------------------------------------------------------
    # APPEL MISTRAL
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    return response


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "🤖 Chatbot RAG — Événements dans le monde"
    )

    print(
        "=========================================="
    )

    print(
        "Tape 'exit' ou 'quit' pour quitter.\n"
    )


    while True:

        question = input(
            "🧑‍💻 Ta question : "
        ).strip()


        # ----------------------------------------------------
        # SORTIE
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nAu revoir 👋"
            )

            break


        # ----------------------------------------------------
        # QUESTION VIDE
        # ----------------------------------------------------

        if not question:

            print(
                "⚠️ Veuillez entrer une question."
            )

            continue


        # ----------------------------------------------------
        # CHATBOT
        # ----------------------------------------------------

        try:

            answer = ask_chatbot(
                question
            ).content

            print(
                "\n🤖 Réponse :\n"
            )

            print(
                answer
            )


        except Exception as e:

            print(
                "\n❌ Une erreur est survenue :"
            )

            print(
                str(e)
            )


        print(
            "\n" + "-" * 60 + "\n"
        )
