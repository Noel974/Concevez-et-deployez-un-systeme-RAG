import pandas as pd
import faiss
import numpy as np
import os


VECTORS_FILE = "data/processed/events_vectors.pkl"
INDEX_FILE = "data/processed/faiss_index.bin"
META_FILE = "data/processed/events_with_vectors.pkl"


# Charger les embeddings
df = pd.read_pickle(VECTORS_FILE)

print("Nombre de vecteurs :", len(df))


# Transformer en matrice numpy
vectors = np.vstack(
    df["vector"].values
).astype("float32")


# Création index FAISS
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)


# Ajouter les vecteurs
index.add(vectors)


# Sauvegarder index
faiss.write_index(
    index,
    INDEX_FILE
)


# Sauvegarder métadonnées
df.to_pickle(
    META_FILE
)


print("----------------------------")
print("Index FAISS créé")
print("Nombre de vecteurs FAISS :", index.ntotal)
print("Index :", INDEX_FILE)
print("Métadonnées :", META_FILE)
print("----------------------------")