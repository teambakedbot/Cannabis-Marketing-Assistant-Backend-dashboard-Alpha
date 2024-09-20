import os
import firebase_admin
from firebase_admin import credentials, firestore
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

# Pinecone initialization
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
embedding_dimension = 3072


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
                response = client.embeddings.create(
                    input=batch_texts, model="text-embedding-3-large"
                )
                print
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                break  # Break if successful
            except openai.OpenAIError as e:
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
        index = pc.Index("product-index")

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
            text = f"{product_data.get('product_name', '')} {product_data.get('raw_product_name', '')} {product_data.get('raw_product_category', '')}"
            texts.append(text)
            # Prepare metadata
            metadata = {
                "sku": product_data.get("cann_sku_id", "") or "",
                "product_name": product_data.get("product_name", "") or "",
                "brand_name": product_data.get("brand_name", "") or "",
                "category": product_data.get("category", "") or "",
                "subcategory": product_data.get("subcategory", "") or "",
                "latest_price": product_data.get("latest_price", 0) or 0,
                "percentage_of_thc": product_data.get("percentage_thc", 0) or 0,
                "percentage_of_cbd": product_data.get("percentage_cbd", 0) or 0,
                "mg_of_thc": product_data.get("mg_thc", 0) or 0,
                "mg_of_cbd": product_data.get("mg_cbd", 0) or 0,
                "for_medical_use": product_data.get("medical", False) or False,
                "for_recreational_use": product_data.get("recreational", False)
                or False,
                "retailer_id": product_data.get("retailer_id", 0) or 0,
                "menu_provider": product_data.get("menu_provider", "") or "",
                "last_updated": (
                    product_data.get("last_updated", "").isoformat()
                    if product_data.get("last_updated")
                    else ""
                ),
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


def fetch_and_upsert_retailers():
    """Fetches retailers from Firestore and upserts embeddings into Pinecone."""
    try:
        retailers_ref = db.collection("retailers")
        retailers = [doc for doc in retailers_ref.stream()]
        logger.info(f"Total retailers fetched: {len(retailers)}")
        index = pc.Index("retailer-index")

        if not retailers:
            logger.info("No retailers found.")
            return

        # Prepare data for embedding
        retailer_ids = []
        texts = []
        metadatas = []

        for retailer in retailers:
            retailer_data = retailer.to_dict()
            retailer_id = retailer.id
            retailer_ids.append(retailer_id)
            # Combine relevant fields for embedding
            text = f"{retailer_data.get('dispensary_name', '')} {retailer_data.get('physical_address', '')} {retailer_data.get('city', '')} {retailer_data.get('state', '')}"
            texts.append(text)
            # Prepare metadata
            metadata = {
                "retailer_id": retailer_id,
                "retailer_name": retailer_data.get("dispensary_name", ""),
                "is_active": bool(retailer_data.get("is_active", False)),
                "cann_dispensary_slug": retailer_data.get("cann_dispensary_slug", ""),
                "website_url": retailer_data.get("website_url", ""),
                "contact_phone": retailer_data.get("contact_phone", ""),
                "contact_email": retailer_data.get("contact_email", ""),
                "city": retailer_data.get("city", ""),
                "address": retailer_data.get("physical_address", ""),
                "state": retailer_data.get("state", ""),
                "zip_code": retailer_data.get("zip_code", ""),
                "country": retailer_data.get("country", ""),
                "latitude": float(retailer_data.get("latitude", 0)),
                "longitude": float(retailer_data.get("longitude", 0)),
                "serves_medical_users": bool(
                    retailer_data.get("serves_medical_users", False)
                ),
                "serves_recreational_users": bool(
                    retailer_data.get("serves_recreational_users", False)
                ),
                "last_updated": (
                    retailer_data.get("last_updated", "").isoformat()
                    if retailer_data.get("last_updated")
                    else ""
                ),
            }
            # Ensure all values are not None
            metadata = {k: (v if v is not None else "") for k, v in metadata.items()}
            metadatas.append(metadata)

        # Generate embeddings
        embeddings = generate_embeddings(texts)

        # Prepare vectors for upsert
        vectors = []
        for retailer_id, embedding, metadata in zip(
            retailer_ids, embeddings, metadatas
        ):
            vectors.append((retailer_id, embedding, metadata))

        # Upsert vectors into Pinecone
        batch_size = 100
        for i in tqdm(
            range(0, len(vectors), batch_size),
            desc="Upserting retailer vectors into Pinecone",
        ):
            batch_vectors = vectors[i : i + batch_size]
            ids_batch = [item[0] for item in batch_vectors]
            embeddings_batch = [item[1] for item in batch_vectors]
            metadata_batch = [item[2] for item in batch_vectors]
            try:
                index.upsert(vectors=zip(ids_batch, embeddings_batch, metadata_batch))
            except Exception as e:
                logger.exception(f"Error upserting retailer vectors to Pinecone: {e}")
        logger.info("Retailer data ingestion completed.")
    except Exception as e:
        logger.exception(f"An error occurred during retailer data ingestion: {e}")


if __name__ == "__main__":
    # fetch_and_upsert_products()
    fetch_and_upsert_retailers()
