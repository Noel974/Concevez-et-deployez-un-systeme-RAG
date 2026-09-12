import pandas as pd

def test_context_generation():
    df = pd.read_pickle("data/processed/events_with_vectors.pkl")
    row = df.iloc[0]

    context = f"""
### Événement : {row['title_fr']}
- Ville : {row['location_city']}
- Catégorie : {row['category']}
- Date : {row['firstdate_begin']}
- Description : {row['description_fr']}
"""

    assert "Événement" in context
    assert row["title_fr"] in context
