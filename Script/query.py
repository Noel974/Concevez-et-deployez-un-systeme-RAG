"""
Chatbot RAG pour les événements.

Pipeline :

Question utilisateur
        ↓
Détection éventuelle de la ville
        ↓
Embedding de la question
        ↓
Recherche FAISS
        ↓
Filtrage par ville
        ↓
Construction du contexte
        ↓
Mistral
        ↓
Réponse
"""

import os
import re

import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings


# ============================================================
# 1. ENVIRONNEMENT
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY est absente du fichier .env"
    )


# ============================================================
# 2. CHEMINS
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INDEX_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "faiss_index.bin"
)

META_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "events_with_vectors.pkl"
)


# ============================================================
# 3. VÉRIFICATIONS
# ============================================================

if not os.path.exists(INDEX_PATH):

    raise FileNotFoundError(
        f"Index FAISS introuvable : {INDEX_PATH}\n"
        "Lance d'abord : python Script/index.py"
    )


if not os.path.exists(META_PATH):

    raise FileNotFoundError(
        f"Métadonnées introuvables : {META_PATH}\n"
        "Lance d'abord : python Script/index.py"
    )


# ============================================================
# 4. CHARGEMENT FAISS
# ============================================================

print("Chargement de l'index FAISS...")

index = faiss.read_index("data/processed/faiss_index.bin")

# --- Charger les métadonnées ---
df = pd.read_pickle("data/processed/events_with_vectors.pkl")

print(
    "Nombre de vecteurs FAISS :",
    index.ntotal
)


# ============================================================
# 5. CHARGEMENT DES MÉTADONNÉES
# ============================================================

df = pd.read_pickle(
    META_PATH
)

print(
    "Nombre d'événements :",
    len(df)
)


# ============================================================
# 6. EMBEDDINGS
# ============================================================

embedder = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=MISTRAL_API_KEY
)


# ============================================================
# 7. MODÈLE MISTRAL
# ============================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0
)


# ============================================================
# 8. NORMALISER LES VILLES
# ============================================================

def normalize_text(text):
    """
    Normalise un texte pour faciliter les comparaisons.
    """

    text = str(text).lower().strip()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# 9. DÉTECTER LA VILLE
# ============================================================

def detect_city(question):
    """
    Cherche une ville présente dans les données
    directement dans la question.
    """

    question_normalized = normalize_text(
        question
    )

    cities = (
        df["location_city"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Trier par longueur décroissante
    # pour tester les noms les plus précis d'abord
    cities = sorted(
        cities,
        key=len,
        reverse=True
    )

    for city in cities:

        city_normalized = normalize_text(
            city
        )

        if city_normalized in question_normalized:

            return city

    return None


# ============================================================
# 10. RECHERCHE FAISS
# ============================================================

def search_events(
    query,
    k=5
):

    """
    Recherche les événements les plus proches
    avec FAISS.
    """

    if index.ntotal == 0:

        return pd.DataFrame()

    # Ne jamais demander plus de résultats
    # que le nombre de vecteurs disponibles
    k = min(
        k,
        index.ntotal
    )

    # --------------------------------------------------------
    # Embedding de la question
    # --------------------------------------------------------

    q_vec = embedder.embed_query(
        query
    )

    q_vec = np.array(
        q_vec,
        dtype="float32"
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Recherche FAISS
    # --------------------------------------------------------

    distances, indices = index.search(
        q_vec,
        k
    )

    valid_indices = [
        i
        for i in indices[0]
        if i >= 0
    ]

    if not valid_indices:

        return pd.DataFrame()

    results = df.iloc[
        valid_indices
    ].copy()

    results["distance"] = distances[0][
        :len(results)
    ]

    return results


# ============================================================
# 11. FILTRAGE PAR VILLE
# ============================================================

def filter_by_city(
    results,
    city
):

    """
    Filtre les résultats sur la ville demandée.
    """

    if city is None:
        return results

    city_normalized = normalize_text(
        city
    )

    mask = results[
        "location_city"
    ].fillna("").apply(
        lambda x:
        normalize_text(x) == city_normalized
    )

    filtered = results[
        mask
    ].copy()

    return filtered


# ============================================================
# 12. CONSTRUIRE LE CONTEXTE
# ============================================================

def build_context(results):

    if results.empty:

        return (
            "Aucun événement correspondant "
            "n'a été trouvé."
        )

    context = ""

    for _, row in results.iterrows():

        context += f"""
### Événement

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

"""

    return context


# ============================================================
# 13. APPEL MISTRAL
# ============================================================

def generate_answer(
    prompt
):

    """
    Appelle Mistral une seule fois.

    Si l'API retourne 429, on affiche une erreur
    claire au lieu d'attendre plusieurs secondes.
    """

    try:

        response = llm.invoke(
            prompt
        )

        return response

    except Exception as e:

        error_message = str(e)

        if (
            "429" in error_message
            or
            "Rate limit" in error_message
            or
            "rate_limited" in error_message
        ):

            print(
                "\n❌ L'API Mistral retourne une erreur 429."
            )

            print(
                "La recherche FAISS fonctionne, "
                "mais la génération Mistral est temporairement limitée."
            )

            print(
                "Vérifie Usage / Limits sur ton compte Mistral."
            )

            return None

        raise


# ============================================================
# 14. CHATBOT
# ============================================================

def ask_chatbot(
    question
):

    # --------------------------------------------------------
    # Détecter la ville
    # --------------------------------------------------------

    city = detect_city(
        question
    )

    if city:

        print(
            f"\n📍 Ville détectée : {city}"
        )

    else:

        print(
            "\n📍 Aucune ville détectée"
        )

    # --------------------------------------------------------
    # Recherche FAISS
    # --------------------------------------------------------

    results = search_events(
        question,
        k=5
    )

    print(
        f"🔎 Résultats FAISS : {len(results)}"
    )

    # --------------------------------------------------------
    # Filtrer par ville
    # --------------------------------------------------------

    if city:

        filtered_results = filter_by_city(
            results,
            city
        )

        # Si FAISS n'a pas ramené les événements
        # de la ville demandée, chercher directement
        # dans la base.

        if filtered_results.empty:

            print(
                "⚠️ FAISS n'a trouvé aucun résultat "
                "dans cette ville."
            )

            direct_results = df[
                df["location_city"]
                .fillna("")
                .apply(
                    lambda x:
                    normalize_text(x)
                    ==
                    normalize_text(city)
                )
            ].copy()

            results = direct_results

        else:

            results = filtered_results

    # --------------------------------------------------------
    # Affichage final
    # --------------------------------------------------------

    print(
        f"📚 Événements retenus : {len(results)}"
    )

    for _, row in results.iterrows():

        print(
            f"  - {row.get('title_fr', '')}"
            f" ({row.get('location_city', '')})"
        )

    # --------------------------------------------------------
    # Aucun événement
    # --------------------------------------------------------

    if results.empty:

        return None

    # --------------------------------------------------------
    # Contexte
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Afficher le contexte pour debug
    # --------------------------------------------------------

    print(
        "\n📖 CONTEXTE RAG :"
    )

    print(
        context
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
Tu es un assistant spécialisé dans les événements.

Tu réponds uniquement à partir du CONTEXTE.

Ne crée aucune information qui n'est pas présente
dans le contexte.

Si plusieurs événements sont présents,
présente-les tous de manière claire.

Si une ville est demandée, ne parle que des
événements de cette ville.

CONTEXTE :

{context}

QUESTION :

{question}

RÉPONSE :
"""

    # --------------------------------------------------------
    # Mistral
    # --------------------------------------------------------

    return generate_answer(
        prompt
    )


# ============================================================
# 15. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "🤖 Chatbot RAG — Événements"
    )

    print(
        "========================================"
    )

    print(
        "Tape 'exit' pour quitter.\n"
    )

    while True:

        question = input(
            "🧑‍💻 Ta question : "
        ).strip()

        # ----------------------------------------------------
        # Quitter
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "Au revoir 👋"
            )

            break

        # ----------------------------------------------------
        # Question vide
        # ----------------------------------------------------

        if not question:

            print(
                "⚠️ Pose une question."
            )

            continue

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        try:

            answer = ask_chatbot(
                question
            )

            if answer is not None:

                print(
                    "\n🤖 Réponse :\n"
                )

                print(
                    answer.content
                )

        except Exception as e:

            print(
                "\n❌ Erreur :"
            )

            print(
                e
            )

        print(
            "\n" + "-" * 60 + "\n"
        )
