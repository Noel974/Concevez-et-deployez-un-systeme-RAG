import pandas as pd
import faiss

def test_metadata_consistency():
    df = pd.read_pickle("data/processed/events_with_vectors.pkl")
    index = faiss.read_index("data/processed/faiss_index.bin")
    assert len(df) == index.ntotal
