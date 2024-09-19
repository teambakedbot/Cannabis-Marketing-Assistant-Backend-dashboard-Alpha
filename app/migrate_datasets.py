import os
import openai
import pinecone
from llama_index import SimpleDirectoryReader
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI initialization
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pinecone initialization
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "knowledge-index"
embedding_dimension = 1536  # For 'text-embedding-ada-002'

# Create Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=embedding_dimension)
index = pinecone.Index(index_name)


def migrate_dataset_to_pinecone(dataset_path, namespace):
    """Migrates a dataset to Pinecone within a specified namespace."""
    # Load documents
    documents = SimpleDirectoryReader(dataset_path).load_data()
    if not documents:
        logger.info(f"No documents found in {dataset_path}")
        return

    logger.info(f"Number of documents in {namespace}: {len(documents)}")

    texts = []
    metadatas = []
    ids = []

    for idx, doc in enumerate(documents):
        text = doc.text
        texts.append(text)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "source": doc.metadata.get("source", ""),
            # Add other metadata fields as needed
        }
        metadatas.append(metadata)
        ids.append(f"{namespace}-{idx}")

    # Generate embeddings
    embeddings = []
    batch_size = 100
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Generating embeddings for {namespace}"
    ):
        batch_texts = texts[i : i + batch_size]
        response = openai.Embedding.create(
            input=batch_texts, model="text-embedding-ada-002"
        )
        batch_embeddings = [data["embedding"] for data in response["data"]]
        embeddings.extend(batch_embeddings)

    # Upsert into Pinecone with namespace
    vectors = []
    for id, embedding, metadata in zip(ids, embeddings, metadatas):
        vectors.append({"id": id, "values": embedding, "metadata": metadata})

    for i in tqdm(
        range(0, len(vectors), batch_size), desc=f"Upserting vectors for {namespace}"
    ):
        batch_vectors = vectors[i : i + batch_size]
        index.upsert(vectors=batch_vectors, namespace=namespace)
    logger.info(f"Completed upserting vectors for {namespace}")


if __name__ == "__main__":
    datasets = {
        "compliance_guidelines": "data/Compliance guidelines",
        "marketing_strategies": "data/Marketing strategies and best practices",
        "seasonal_marketing": "data/Seasonal and holiday marketing plans",
        "state_policies": "data/State-specific cannabis marketing regulations",
    }

    for namespace, path in datasets.items():
        migrate_dataset_to_pinecone(dataset_path=path, namespace=namespace)
