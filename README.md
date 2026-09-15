"# Concevez-et-deployez-un-systeme-RAG" 
# Concevoir et déployer un systeme RAG 

## 👤 Auteur

Projet réalisé par **Noel Emmanuel**

* GitHub : https://github.com/Noel974
* LinkedIn : https://www.linkedin.com/in/Antoine-Noel/
* Email : [noelantoine974@outlook.fr](mailto:noelantoine974@outlook.fr)

---

# 📋 Sommaire

* [Introduction](#introduction)
* [Technologies utilisées](#technologies-utilisées)
* [Réalisation](#realisation)

---
Système de **Retrieval-Augmented Generation (RAG)** qui collecte des données d'événements culturels via une API publique, les vectorise, les indexe avec FAISS, puis permet à un utilisateur de poser des questions en langage naturel et d'obtenir des recommandations d'événements générées par un LLM (Mistral) à partir des données réellement trouvées dans la base.

## Introduction 
  Le but est de construire un assistant conversationnel capable de répondre à des questions du type *"Quels concerts y a-t-il à Lyon ce mois-ci ?"* en s'appuyant uniquement sur des événements réels récupérés depuis une API (type OpenAgenda / OpenDataSoft), plutôt que sur les connaissances générales du LLM. C'est le principe du RAG : **on ne laisse pas le modèle inventer, on le force à répondre à partir d'un contexte récupéré (retrieval) dans une base vectorielle.**

## Technologie 

python 
faiss
mistral 
langchain
huggingface

installtation des depance 
```bash
pip install langchain faiss-cpu langchain-mistralai langchain-huggingface sentence-transformers

```

## Réalisation
Le projet fonctionne en 4 étapes séquentielles, chacune correspondant à un script :

```
flowchart TD

    subgraph Collecte & Prétraitement
        A[agenda.py\nCollecte API\nFiltrage dates\nDéduplication par uid]
    end

    subgraph Vectorisation
        B[embed.py\nAppels Mistral\nGénération embeddings\nMise à jour events_vectors.pkl]
        U[utils/mistral.py\nClient MistralAIEmbeddings]
    end

    subgraph Indexation
        C[index.py\nConstruction index FAISS\nSynchronisation vecteurs + métadonnées]
    end

    subgraph Chatbot RAG
        D[query.py\nsearch_events_smart()\ndetect_city()\nbuild_context()\nask_chatbot()]
    end

    subgraph Données
        CSV[(events.csv)]
        PKL1[(events_vectors.pkl)]
        PKL2[(events_with_vectors.pkl)]
        FAISS[(faiss_index.bin)]
    end

    %% Flux de données
    A --> CSV
    A --> B
    B --> PKL1
    B --> PKL2
    C --> FAISS
    PKL1 --> C
    PKL2 --> D
    FAISS --> D

    %% Dépendances externes
    API[(API OpenAgenda)]
    MistralE[(Mistral Embed API)]
    MistralLLM[(Mistral LLM API)]

    API --> A
    U --> B
    MistralE --> B
    MistralLLM --> D

    %% Utilisateur
    User([Utilisateur])
    User --> D

```

1. **`aganda.py`** interroge l'API et sauvegarde les événements bruts dans `data/processed/events.csv`.
2. **`embed.py`** lit ce CSV, calcule un embedding pour chaque nouvel événement, et sauvegarde le tout dans `data/processed/events_vectors.pkl`.
3. **`index.py`** charge ces vecteurs, construit un index FAISS (recherche par similarité) et sauvegarde l'index (`faiss_index.bin`) ainsi que les métadonnées associées (`events_with_vectors.pkl`).
4. **`query.py`** est le chatbot final : il transforme la question de l'utilisateur en vecteur, cherche les événements les plus proches dans FAISS, construit un contexte textuel, puis l'envoie à un LLM Mistral qui rédige la réponse.

Le pipeline est **incrémental** : à chaque exécution de `aganda.py` et `embed.py`, seuls les nouveaux événements (identifiés par leur `uid`) sont ajoutés, sans dupliquer ni retraiter tout l'historique.

### Mise en route 
Création de l'environnement virtuel de pyton 
```bash
python -m venv rag-venv
```
Activation de l'environnement 
```bash
.rag-venv\Scripts\activate
``` 
d'installer les libraries 
pandas — traitement des données

requests — appels API

faiss-cpu — base vectorielle

langchain — orchestration LLM

langchain-mistralai — client Mistral

sentence-transformers — alternative embeddings

pytest — tests unitaires

python-dotenv — gestion des variables d’environnement

installtation des depance 
```bash
pip install pandas requests faiss-cpu langchain langchain-mistralai sentence-transformers python-dotenv pytest

```
ou bien de utiliser requirement.txt 
```bash 
pip install -r requirements.txt

```
#### Deroulement 
Création des scripts test , index, aganda, query , embed
Le script test permet de faire un test 

un dossier test qui permet de tester le systeme rag 

$env:TEST_REGION="Saint-Denis"; pytest -v


---

## 3. Description des fichiers

| Fichier | Rôle |
|---|---|
| **`aganda.py`** | Récupère les événements depuis une API (région/ville saisies par l'utilisateur), nettoie les dates, filtre sur une période (`DAYS_HISTORY`, 365 jours par défaut), construit un champ `text_for_embedding` (titre + description + ville + date), et fusionne avec l'historique existant (`data/processed/events.csv`) en supprimant les doublons par `uid`. |
| **`embed.py`** | Charge `events.csv`, compare avec les vecteurs déjà calculés (`events_vectors.pkl`), et n'envoie à l'embedder (`utils/mistral.py` → fonction `embed_texts`) que les événements réellement nouveaux, avant de fusionner et sauvegarder. |
| **`index.py`** | Charge les vecteurs, les empile dans une matrice NumPy, crée un index FAISS de type `IndexFlatL2` (recherche exacte par distance euclidienne) et le sauvegarde avec les métadonnées correspondantes. |
| **`query.py`** | Point d'entrée du chatbot. Charge l'index FAISS et les métadonnées, encode la question utilisateur avec `MistralAIEmbeddings` (`mistral-embed`), récupère les `k=3` événements les plus proches, construit un prompt contextuel, et interroge `ChatMistralAI` (`mistral-large-latest`) pour générer la réponse finale. Boucle interactive en console (`exit` pour quitter). |
| **`test.py`** | Script de vérification de l'environnement : teste les imports (FAISS, LangChain), vérifie si le support GPU FAISS est actif, teste un embedding avec HuggingFace (`all-MiniLM-L6-v2`) et un appel simple à `ChatMistralAI`. **À usage de diagnostic uniquement**, ne fait pas partie du pipeline de production. |
| **`utils/mistral.py`** | Contient la fonction `embed_texts(texts: list[str]) -> list[list[float]]` : instancie `MistralAIEmbeddings("mistral-embed")` et calcule un vecteur pour chaque texte via `embed_query`, appelé en boucle depuis `embed.py`. |

---

## 4. Prérequis et installation

### Dépendances Python
```bash
pip install pandas requests python-dotenv faiss-cpu numpy langchain-mistralai langchain-huggingface
```

### Variables d'environnement (`.env`)
```env
API_BASE_URL=<url_de_l_api_evenements>
DAYS_HISTORY=365
MISTRAL_API_KEY=<votre_clé_api_mistral>
```

> ⚠️ **Important** : ne jamais coder une clé API en dur dans le code source (voir `test.py` actuellement). Utilisez systématiquement `os.getenv(...)`, et régénérez toute clé qui aurait été exposée par erreur.

### Structure des dossiers attendue
```
projet/
├── .env
├── aganda.py
├── embed.py
├── index.py
├── query.py
├── test.py
├── utils/
│   └── mistral.py
└── data/
    └── processed/
        ├── events.csv
        ├── events_vectors.pkl
        ├── events_with_vectors.pkl
        └── faiss_index.bin
```

---

## 5. Utilisation

Exécuter les scripts **dans l'ordre**, à chaque mise à jour des données :

```bash
# 1. Récupérer les événements (demande une région et une ville en console)
python aganda.py

# 2. Calculer les embeddings des nouveaux événements
python embed.py

# 3. (Re)construire l'index FAISS
python index.py

# 4. Lancer le chatbot
python query.py
```

Exemple d'interaction avec `query.py` :
```
🧑‍💻 Ta question : Quels événements culturels à Lyon ce week-end ?

🤖 Réponse :
D'après les événements disponibles...
```
