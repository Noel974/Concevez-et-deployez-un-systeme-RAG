from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
import faiss

import os
from dotenv import load_dotenv

load_dotenv()


print("Imports OK")

# Vérifier Faiss CPU
print("Faiss version:", faiss.__version__)
print("Faiss GPU support:", hasattr(faiss, "StandardGpuResources"))

# Test embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = emb.embed_query("Test embedding")
print("Embedding size:", len(vec))

# Test Mistral
chat = ChatMistralAI(
    model="mistral-small-latest",
    api_key = os.getenv("MISTRAL_API_KEY")
)

resp = chat.invoke("Réponds simplement : OK.")
print(resp)