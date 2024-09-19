import os
import firebase_admin
from firebase_admin import credentials, firestore
import openai
import pinecone
from tqdm import tqdm
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv(override=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase initialization
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)
db = firestore.client()

# OpenAI initialization
openai.api_key = os.getenv("OPENAI_API_KEY")

print("$$$", os.environ.get("PINECONE_API_KEY"))

# Pinecone initialization
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "product-index"
embedding_dimension = 3072  # For 'text-embedding-3-large'

index = pc.Index(index_name)


def generate_embeddings(texts):
    """Generates embeddings for a list of texts using OpenAI's API with retry logic."""
    embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = openai.Embedding.create(
                    input=batch_texts, model="text-embedding-3-large"
                )
                batch_embeddings = [data["embedding"] for data in response["data"]]
                embeddings.extend(batch_embeddings)
                break  # Break if successful
            except openai.error.OpenAIError as e:
                retries += 1
                wait_time = 2**retries
                logger.warning(
                    f"OpenAI API error: {e}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                break  # Exit the retry loop for unexpected errors
        else:
            logger.error("Max retries exceeded while generating embeddings.")
            raise RuntimeError("Failed to generate embeddings after multiple retries.")
    return embeddings


def fetch_and_upsert_products():
    """Fetches products from Firestore and upserts embeddings into Pinecone."""
    try:
        products_ref = db.collection("products")
        products = [doc for doc in products_ref.stream()]
        logger.info(f"Total products fetched: {len(products)}")

        if not products:
            logger.info("No products found.")
            return

        # Prepare data for embedding
        product_ids = []
        texts = []
        metadatas = []

        for product in products:
            product_data = product.to_dict()
            product_id = product.id
            product_ids.append(product_id)
            # Combine relevant fields for embedding
            text = f"{product_data.get('name', '')} {product_data.get('description', '')} {product_data.get('category', '')}"
            texts.append(text)
            # Prepare metadata
            metadata = {
                "name": product_data.get("name", ""),
                "category": product_data.get("category", ""),
                "brand": product_data.get("brand_name", ""),
                "price": product_data.get("latest_price", 0),
                # Add more metadata fields as needed
            }
            metadatas.append(metadata)

        # Generate embeddings
        embeddings = generate_embeddings(texts)

        # Prepare vectors for upsert
        vectors = []
        for product_id, embedding, metadata in zip(product_ids, embeddings, metadatas):
            vectors.append((product_id, embedding, metadata))

        # Upsert vectors into Pinecone
        batch_size = 100
        for i in tqdm(
            range(0, len(vectors), batch_size), desc="Upserting vectors into Pinecone"
        ):
            batch_vectors = vectors[i : i + batch_size]
            ids_batch = [item[0] for item in batch_vectors]
            embeddings_batch = [item[1] for item in batch_vectors]
            metadata_batch = [item[2] for item in batch_vectors]
            try:
                index.upsert(vectors=zip(ids_batch, embeddings_batch, metadata_batch))
            except Exception as e:
                logger.exception(f"Error upserting vectors to Pinecone: {e}")
        logger.info("Data ingestion completed.")
    except Exception as e:
        logger.exception(f"An error occurred during data ingestion: {e}")


if __name__ == "__main__":
    fetch_and_upsert_products()
