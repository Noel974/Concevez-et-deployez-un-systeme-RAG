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
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

try:
    resp = chat.invoke("Réponds simplement : OK.")
    print("Mistral OK :", resp.content)

except Exception as e:
    print("❌ Mistral ERREUR :", e)
    print("💡 Conseil : tu as probablement dépassé le rate limit. Attends quelques secondes et réessaie.")
