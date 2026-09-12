import os
import pandas as pd
from datetime import datetime, timedelta


VECTORS_FILE = "data/processed/events_with_vectors.pkl"


def test_events_are_recent_and_in_region():

    # ========================================================
    # CONFIGURATION
    # ========================================================

    expected_region = os.getenv("TEST_REGION")

    if not expected_region:
        raise ValueError(
            "⚠️ TEST_REGION n'est pas définie.\n"
            "Exemple PowerShell :\n"
            '$env:TEST_REGION="Réunion"'
        )

    # ========================================================
    # VÉRIFICATION DU FICHIER
    # ========================================================

    assert os.path.exists(VECTORS_FILE), (
        "❌ events_with_vectors.pkl introuvable."
    )

    # ========================================================
    # CHARGEMENT
    # ========================================================

    df = pd.read_pickle(VECTORS_FILE)

    assert len(df) > 0, (
        "❌ Aucun événement dans la base vectorielle."
    )

    print(
        f"\n📊 Nombre d'événements : {len(df)}"
    )

    # ========================================================
    # VÉRIFICATION DES COLONNES
    # ========================================================

    required_columns = [
        "firstdate_begin",
        "location_city"
    ]

    for column in required_columns:

        assert column in df.columns, (
            f"❌ Colonne '{column}' absente."
        )

    # ========================================================
    # VÉRIFICATION DES DATES
    # ========================================================

    df["firstdate_begin"] = pd.to_datetime(
        df["firstdate_begin"],
        errors="coerce",
        format="mixed",    
        utc=True
        ).dt.tz_localize(None)


    # --------------------------------------------------------
    # Vérifier les dates invalides
    # --------------------------------------------------------

    invalid_dates = df[
        df["firstdate_begin"].isna()
    ]

    assert invalid_dates.empty, (
        "❌ Des événements possèdent une date invalide.\n"
        f"Nombre de dates invalides : {len(invalid_dates)}\n"
        f"Indices concernés : {invalid_dates.index.tolist()}"
    )

    # ========================================================
    # VÉRIFICATION DE L'ANCIENNETÉ
    # ========================================================

    one_year_ago = (
        datetime.now()
        - timedelta(days=365)
    )

    old_events = df[
        df["firstdate_begin"] < one_year_ago
    ]

    assert old_events.empty, (
        "❌ Certains événements ont plus d'un an.\n"
        f"Nombre d'événements trop anciens : "
        f"{len(old_events)}"
    )

    # ========================================================
    # VÉRIFICATION DES LOCALISATIONS
    # ========================================================

    cities = (
        df["location_city"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cities = cities[
        cities != ""
    ]

    assert len(cities) > 0, (
        "❌ Aucune ville trouvée."
    )

    print(
        "\n🏙️ Villes présentes :"
    )

    print(
        cities.unique()
    )

    # ========================================================
    # CAS TEST_REGION = RÉUNION
    # ========================================================

    if expected_region.lower() in [
        "réunion",
        "reunion"
    ]:

        # ----------------------------------------------------
        # Villes principales de La Réunion
        # ----------------------------------------------------

        reunion_cities = [
            "Saint-Denis",
            "Saint-Pierre",
            "Saint-Paul",
            "Le Tampon",
            "Saint-André",
            "Saint-Louis",
            "Saint-Joseph",
            "Saint-Benoît"
        ]

        # ----------------------------------------------------
        # Recherche des villes présentes
        # ----------------------------------------------------

        found_cities = [
            city
            for city in reunion_cities
            if city.lower()
            in [
                c.lower()
                for c in cities.unique()
            ]
        ]

        assert len(found_cities) > 0, (
            "❌ Aucun événement de La Réunion "
            "n'a été trouvé dans la base.\n"
            f"Villes trouvées : "
            f"{list(cities.unique())}"
        )

        print(
            "\n✅ Région 'Réunion' validée."
        )

        print(
            "Villes de La Réunion trouvées :",
            found_cities
        )

        return

    # ========================================================
    # CAS TEST_REGION = UNE VILLE
    # ========================================================

    expected_city = expected_region.strip()

    matching_cities = [
        city
        for city in cities.unique()
        if city.lower() == expected_city.lower()
    ]

    assert len(matching_cities) > 0, (
        f"❌ Aucun événement ne correspond "
        f"à '{expected_city}'.\n"
        f"Localisations trouvées : "
        f"{list(cities.unique())}"
    )

    print(
        f"\n✅ Ville '{expected_city}' trouvée."
    )
