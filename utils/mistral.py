"""
Module utilitaire pour la vectorisation de textes via MistralAI.

Fonctionnalités :
- Chargement automatique de la clé API depuis le fichier .env
- Utilisation du modèle 'mistral-embed' pour générer des embeddings
- Fonction embed_texts() utilisée dans embed.py pour vectoriser les événements

Ce module constitue une brique essentielle du pipeline RAG :
1. agenda.py : ingestion & pré-processing
2. embed.py : vectorisation via embed_texts()
3. index.py : création de l’index FAISS
"""

from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Génère les embeddings pour une liste de textes en utilisant MistralAI.

    Args:
        texts (list[str]): Liste de textes à vectoriser.

    Returns:
        list[list[float]]: Liste de vecteurs (embeddings) correspondant aux textes.
    """
    emb = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )

    return [emb.embed_query(t) for t in texts]
