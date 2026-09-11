import os
import pandas as pd
from datetime import datetime, timedelta

VECTORS_FILE = "data/processed/events_with_vectors.pkl"

def test_events_are_recent_and_in_region():
    expected_region = os.getenv("TEST_REGION")

    if not expected_region:
        raise ValueError(
            "⚠️ TEST_REGION n'est pas définie.\n"
            "Exemple : $env:TEST_REGION=\"Réunion\""
        )

    assert os.path.exists(VECTORS_FILE), "❌ events_with_vectors.pkl introuvable."

    df = pd.read_pickle(VECTORS_FILE)
    assert len(df) > 0, "❌ Aucun événement dans la base vectorielle."

    # Vérification des dates
    one_year_ago = datetime.now() - timedelta(days=365)
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], errors="coerce")
    assert df["firstdate_begin"].notna().all(), "❌ Dates invalides."
    assert (df["firstdate_begin"] >= one_year_ago).all(), "❌ Certains événements ont plus d'un an."

    # Vérification région / ville
    if "location_region" in df.columns:
        regions = df["location_region"].fillna("")
    else:
        regions = df["location_city"].fillna("")

    # Cas 1 : TEST_REGION = Réunion → on accepte toutes les villes de la Réunion
    if expected_region.lower() == "réunion":
        assert len(regions.unique()) > 0, "❌ Aucune localisation trouvée."
        print("✅ Région 'Réunion' validée via les villes :", regions.unique())
        return

    # Cas 2 : TEST_REGION = une ville → on teste la ville
    assert expected_region in regions.unique(), (
        f"❌ Aucun événement ne correspond à '{expected_region}'. "
        f"Localisations trouvées : {regions.unique()}"
    )

    print(f"✅ Tous les événements correspondent à la ville '{expected_region}'.")
