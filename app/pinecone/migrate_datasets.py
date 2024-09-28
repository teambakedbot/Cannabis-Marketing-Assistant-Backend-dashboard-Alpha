import json
import os
from openai import OpenAI

from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv(override=True)

# Initialize logging
from ..config.config import logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "knowledge-index"

index = pc.Index(index_name)


def generate_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return response.data[0].embedding


def migrate_docstore_to_pinecone(docstore_path, namespace):
    # Load the docstore JSON file
    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    doc_data = docstore.get("docstore/data", {})
    if not doc_data:
        logger.info(f"No document data found in {docstore_path}")
        return

    vectors = []
    batch_size = 100

    for doc_id, doc_info in tqdm(doc_data.items(), desc="Processing Documents"):
        doc_content = doc_info.get("__data__", {})
        text = doc_content.get("text", "")
        metadata = doc_content.get("metadata", {})
        if not text:
            continue

        # Generate embedding
        embedding = generate_embedding(text)

        # Include the text in metadata
        metadata["text"] = text

        # Prepare vector data
        vector = {"id": doc_id, "values": embedding, "metadata": metadata}
        vectors.append(vector)

        # Upsert in batches
        if len(vectors) >= batch_size:
            upsert_vectors(vectors, namespace)
            vectors = []

    # Upsert any remaining vectors
    if vectors:
        upsert_vectors(vectors, namespace)
        vectors = []

    logger.info(f"Completed migration of {docstore_path} to Pinecone.")


def upsert_vectors(vectors, namespace):
    index.upsert(vectors=vectors, namespace=namespace)


if __name__ == "__main__":
    datasets = {
        "compliance_guidelines": "app/data/Compliance guidelines/docstore.json",
        "marketing_strategies": "app/data/Marketing strategies and best practices/docstore.json",
        "seasonal_marketing": "app/data/Seasonal and holiday marketing plans/docstore.json",
        "state_policies": "app/data/State-specific cannabis marketing regulations/docstore.json",
    }

    for namespace, docstore_file in datasets.items():
        logger.info(f"Starting migration for namespace: {namespace}")
        migrate_docstore_to_pinecone(docstore_path=docstore_file, namespace=namespace)
