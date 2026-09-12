import os
import faiss

def test_faiss_index_exists():
    assert os.path.exists("data/processed/faiss_index.bin")

def test_faiss_index_load():
    index = faiss.read_index("data/processed/faiss_index.bin")
    assert index.ntotal > 0
