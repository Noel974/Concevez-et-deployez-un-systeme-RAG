import os
import pandas as pd

def test_region():
    # Récupération de la région attendue
    expected_region = os.getenv("TEST_REGION")
    if not expected_region:
        raise ValueError("TEST_REGION doit être définie")

    expected_region = expected_region.lower()

    # Chargement des données vectorisées
    df = pd.read_pickle("data/processed/events_with_vectors.pkl")

    # Extraction des villes présentes dans les données
    locations = df["location_city"].str.lower().unique()

    # Cas particulier : La Réunion = plusieurs villes possibles
    if expected_region == "réunion":
        assert len(locations) > 0, (
            "Aucune ville trouvée dans les données pour la région 'Réunion'."
        )
        return

    # Cas général : la région est une ville unique
    assert expected_region in locations, (
        f"La ville '{expected_region}' n'est pas présente dans les données.\n"
        f"Villes disponibles : {locations}"
    )
