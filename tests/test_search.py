import numpy as np
import pandas as pd
import faiss

def test_search_faiss():
    index = faiss.read_index("data/processed/faiss_index.bin")
    df = pd.read_pickle("data/processed/events_with_vectors.pkl")

    dummy_vec = np.zeros((1, index.d), dtype="float32")
    distances, indices = index.search(dummy_vec, 1)

    assert indices[0][0] >= 0
    assert indices[0][0] < len(df)
