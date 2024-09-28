import os
import firebase_admin
from firebase_admin import credentials, firestore
import openai
from openai import OpenAI
from ..config.config import settings


client = OpenAI(api_key=settings.OPENAI_API_KEY)
from tqdm import tqdm
import logging
import time
from pinecone import Pinecone
from ..config.config import logger
from typing import List, Dict
from ..utils.firebase_utils import db
from ..config.config import settings
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)


# OpenAI initialization
# Pinecone initialization
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
embedding_dimension = 3072


def generate_embeddings(texts: List[str]) -> List[List[float]]:
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


async def fetch_and_upsert_collection(
    collection_name: str,
    index_name: str,
    text_fields: List[str],
    metadata_fields: Dict[str, str],
):
    try:
        collection_ref = db.collection(collection_name)
        documents = [doc for doc in await collection_ref.get()]
        logger.info(f"Total {collection_name} fetched: {len(documents)}")
        index = pc.Index(index_name)

        if not documents:
            logger.info(f"No {collection_name} found.")
            return

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_data = doc.to_dict()
            doc_id = doc.id
            ids.append(doc_id)
            text = " ".join([str(doc_data.get(field, "")) for field in text_fields])
            texts.append(text)
            metadata = {}
            for key, field in metadata_fields.items():
                value = doc_data.get(field)
                metadata[key] = process_metadata_value(value)
            metadatas.append(metadata)

        embeddings = generate_embeddings(texts)

        vectors = list(zip(ids, embeddings, metadatas))

        batch_size = 100
        for i in tqdm(
            range(0, len(vectors), batch_size),
            desc=f"Upserting vectors into Pinecone for {collection_name}",
        ):
            batch_vectors = vectors[i : i + batch_size]
            try:
                index.upsert(
                    vectors=zip(
                        [item[0] for item in batch_vectors],
                        [item[1] for item in batch_vectors],
                        [item[2] for item in batch_vectors],
                    )
                )
            except Exception as e:
                logger.exception(f"Error upserting vectors to Pinecone: {e}")
        logger.info(f"{collection_name.capitalize()} data ingestion completed.")
    except Exception as e:
        logger.exception(
            f"An error occurred during {collection_name} data ingestion: {e}"
        )


def process_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def fetch_and_upsert_products():
    text_fields = ["product_name", "raw_product_name", "raw_product_category"]
    metadata_fields = {
        "sku": "cann_sku_id",
        "product_name": "product_name",
        "raw_product_name": "raw_product_name",
        "raw_product_category": "raw_product_category",
        "brand_name": "brand_name",
        "category": "category",
        "subcategory": "subcategory",
        "image_url": "image_url",
        "url": "url",
        "latest_price": "latest_price",
        "percentage_of_thc": "percentage_thc",
        "percentage_of_cbd": "percentage_cbd",
        "mg_of_thc": "mg_thc",
        "mg_of_cbd": "mg_cbd",
        "display_weight": "display_weight",
        "for_medical_use": "medical",
        "for_recreational_use": "recreational",
        "retailer_id": "retailer_id",
        "menu_provider": "menu_provider",
        "updated_at": "updated_at",
    }
    await fetch_and_upsert_collection(
        "products", "product-index", text_fields, metadata_fields
    )


async def fetch_and_upsert_retailers():
    text_fields = ["dispensary_name", "physical_address", "city", "state"]
    metadata_fields = {
        "retailer_id": "retailer_id",
        "retailer_name": "dispensary_name",
        "is_active": "is_active",
        "cann_dispensary_slug": "cann_dispensary_slug",
        "website_url": "website_url",
        "contact_phone": "contact_phone",
        "contact_email": "contact_email",
        "city": "city",
        "address": "physical_address",
        "state": "state",
        "zip_code": "zip_code",
        "country": "country",
        "latitude": "latitude",
        "longitude": "longitude",
        "serves_medical_users": "serves_medical_users",
        "serves_recreational_users": "serves_recreational_users",
        "updated_at": "updated_at",
    }
    await fetch_and_upsert_collection(
        "retailers", "retailer-index", text_fields, metadata_fields
    )


if __name__ == "__main__":
    fetch_and_upsert_products()
    # fetch_and_upsert_retailers()
