import os
import pandas as pd

def test_embeddings_file_exists():
    assert os.path.exists("data/processed/events_vectors.pkl")

def test_embeddings_has_vectors():
    df = pd.read_pickle("data/processed/events_vectors.pkl")
    assert "vector" in df.columns
    assert len(df) > 0

def test_embeddings_vector_format():
    df = pd.read_pickle("data/processed/events_vectors.pkl")
    vec = df.iloc[0]["vector"]
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
