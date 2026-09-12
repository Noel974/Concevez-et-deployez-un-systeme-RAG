import pandas as pd
from datetime import datetime, timedelta


VECTORS_FILE = "data/processed/events_with_vectors.pkl"


def test_dates_recent():

    # ========================================================
    # CHARGEMENT
    # ========================================================

    df = pd.read_pickle(
        VECTORS_FILE
    )

    assert len(df) > 0, (
        "❌ Aucun événement dans events_with_vectors.pkl."
    )

    # ========================================================
    # VÉRIFICATION COLONNE
    # ========================================================

    assert "firstdate_begin" in df.columns, (
        "❌ La colonne 'firstdate_begin' est absente."
    )

    # ========================================================
    # CONVERSION DES DATES
    # ========================================================
    #
    # utc=True permet de gérer les dates avec différents
    # fuseaux horaires.
    #
    # Exemple :
    # 2026-08-11 04:30:00
    # 2026-10-18T12:00:00+00:00
    #
    # Les deux sont converties vers UTC puis le fuseau
    # est retiré pour permettre les comparaisons avec
    # datetime.now().
    # ========================================================

    df["firstdate_begin"] = pd.to_datetime(
        df["firstdate_begin"],
        errors="coerce",
        format="mixed",
        utc=True
    ).dt.tz_localize(None)

    # ========================================================
    # DATES INVALIDES
    # ========================================================

    invalid_dates = df[
        df["firstdate_begin"].isna()
    ]

    assert invalid_dates.empty, (
        "\n❌ Dates invalides détectées.\n"
        f"Nombre : {len(invalid_dates)}\n"
        f"Indices concernés : "
        f"{invalid_dates.index.tolist()}"
    )

    # ========================================================
    # DATE MINIMALE
    # ========================================================

    one_year_ago = (
        datetime.now()
        - timedelta(days=365)
    )

    # ========================================================
    # ÉVÉNEMENTS TROP ANCIENS
    # ========================================================

    old_events = df[
        df["firstdate_begin"] < one_year_ago
    ]

    assert old_events.empty, (
        "\n❌ Certains événements ont plus d'un an.\n"
        f"Nombre : {len(old_events)}\n"
        f"Date minimale autorisée : {one_year_ago}\n"
        f"Indices concernés : "
        f"{old_events.index.tolist()}"
    )
