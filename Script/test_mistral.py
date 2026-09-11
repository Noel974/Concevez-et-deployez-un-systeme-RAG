import os

from dotenv import load_dotenv
from mistralai.client import Mistral


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("❌ MISTRAL_API_KEY introuvable")
    exit()

print("✅ Clé Mistral trouvée")

try:

    client = Mistral(
        api_key=api_key
    )

    print("✅ Client Mistral créé")
    print("Test de l'API...")

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Réponds simplement : OK"
            }
        ]
    )

    print("\n✅ API Mistral fonctionne !")
    print("Réponse :")
    print(
        response.choices[0].message.content
    )

except Exception as e:

    print("\n❌ Erreur Mistral :")
    print(type(e).__name__)
    print(str(e))
