"""
tests/test_query_search.py — Vérifie le comportement réel de la recherche
hybride (Script/query.py) plutôt que la pureté de tout le stockage brut.

Contexte : la base vectorielle héberge volontairement plusieurs villes en
même temps (architecture multi-villes assumée). La règle métier à tester
n'est donc plus "toute la base ne contient qu'une ville", mais :

    "Quand une ville est détectée dans la question, TOUS les événements
     RETOURNÉS par search_events_smart() doivent appartenir à cette ville,
     et dater de moins de DAYS_HISTORY jours."

C'est le comportement que le prompt de ask_chatbot() promet à l'utilisateur
("Ne propose aucun événement situé dans une autre ville") — ce test vérifie
que cette promesse est bien tenue par le code, pas seulement par le prompt.

L'appel réel à l'API Mistral (embedder.embed_query) est remplacé par un
vecteur factice (monkeypatch) : on teste la logique de filtrage de
search_events_smart, pas la qualité sémantique du modèle d'embedding, ce qui
évite de dépendre du réseau et de consommer du quota API à chaque exécution
des tests.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(ROOT_DIR, "Script")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import query as rag  # noqa: E402  (import après modification de sys.path)

DAYS_HISTORY = int(os.getenv("DAYS_HISTORY", 365))


@pytest.fixture
def target_city():
    """
    Utilise une vraie ville présente dans les données chargées, plutôt
    qu'une ville codée en dur — le test reste valable quelle que soit
    la composition actuelle de la base.
    """
    if not rag.CITIES:
        pytest.skip("Aucune ville détectée dans les données chargées.")
    return rag.CITIES[0]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """
    Remplace l'appel réseau à Mistral par un vecteur déterministe (pas d'API, pas de coût).

    Patché au niveau de la CLASSE (pas de l'instance rag.embedder) : MistralAIEmbeddings
    est un modèle Pydantic qui interdit d'assigner un attribut arbitraire directement
    sur une instance existante.
    """
    dimension = rag.index.d

    def fake_embed_query(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(dimension).tolist()

    monkeypatch.setattr(rag.MistralAIEmbeddings, "embed_query", fake_embed_query)


def test_detect_city_finds_known_city(target_city):
    """detect_city doit reconnaître une ville présente dans la base quand elle est citée."""
    question = f"Quels événements à {target_city} ?"
    detected = rag.detect_city(question)
    assert detected is not None
    assert detected.casefold() == target_city.casefold()


def test_detect_city_returns_none_when_no_city_mentioned():
    """detect_city ne doit pas halluciner une ville absente de la question."""
    assert rag.detect_city("Quels événements intéressants en ce moment ?") is None


def test_search_smart_filters_strictly_on_detected_city(target_city):
    """
    Règle métier 1 : quand une ville est détectée, TOUS les résultats
    retournés par search_events_smart doivent appartenir à cette ville —
    même si la base contient d'autres villes.
    """
    question = f"Quels événements à {target_city} ?"
    results = rag.search_events_smart(question, k=5)

    if results.empty:
        pytest.skip(f"Aucun événement retourné pour '{target_city}'.")

    mismatched = results[
        results["location_city"].str.casefold() != target_city.casefold()
    ]

    assert mismatched.empty, (
        f"{len(mismatched)} résultat(s) hors ville détectée '{target_city}' "
        f"ont été retournés par search_events_smart :\n"
        f"{mismatched['location_city'].tolist()}"
    )


def test_search_smart_results_are_recent(target_city):
    """
    Règle métier 2 : tous les événements retournés pour une ville donnée
    doivent dater de moins de DAYS_HISTORY jours.
    """
    question = f"Quels événements à {target_city} ?"
    results = rag.search_events_smart(question, k=5)

    if results.empty:
        pytest.skip(f"Aucun événement retourné pour '{target_city}'.")

    min_date = pd.Timestamp(datetime.now(), tz="UTC") - pd.Timedelta(days=DAYS_HISTORY)
    dates = pd.to_datetime(results["firstdate_begin"], errors="coerce", utc=True)
    too_old = results[dates < min_date]

    assert too_old.empty, (
        f"{len(too_old)} événement(s) retourné(s) pour '{target_city}' "
        f"datent de plus de {DAYS_HISTORY} jours."
    )


def test_search_smart_without_city_does_not_crash():
    """Une question sans ville détectée doit renvoyer une recherche FAISS classique, sans erreur."""
    results = rag.search_events_smart("Quels événements intéressants en ce moment ?", k=5)
    assert isinstance(results, pd.DataFrame)