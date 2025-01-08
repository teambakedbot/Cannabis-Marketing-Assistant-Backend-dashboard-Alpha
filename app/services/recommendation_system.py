import numpy as np
from ..config.config import settings
from ..utils.firebase_utils import db
from openai import OpenAI
from pinecone import Pinecone
from ..models.schemas import Product, GroupedProduct, Pagination, ProductResults
from redis import Redis
import json
from functools import lru_cache
import os
from typing import List
from datetime import datetime
import uuid
from ..pinecone.data_ingestion import fetch_and_upsert_products
from ..config.config import settings
from datetime import timedelta

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("product-index")

# Initialize Redis
redis_url = settings.REDISCLOUD_URL
redis_client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)


def get_text_embedding(text: str) -> list:
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding


@lru_cache(maxsize=1000)
def get_product_embedding(product_data):
    text = f"{product_data.get('name', '')} {product_data.get('category', '')} {product_data.get('description', '')}"
    embedding = get_text_embedding(text)
    return embedding


async def update_embeddings():
    await fetch_and_upsert_products()


@lru_cache(maxsize=1000)
async def get_user_preferences(user_id):
    try:
        interactions = (
            await db.collection("interactions").where("user_id", "==", user_id).get()
        )
        preferences = np.zeros(100)  # Assume 100-dimensional embeddings
        for interaction in interactions:
            product = (
                await db.collection("products").document(interaction.product_id).get()
            )
            embedding = get_product_embedding(product)
            preferences += embedding * interaction.rating

        return preferences / len(interactions) if interactions else preferences
    finally:
        db.close()


async def get_recommendations(user_id, n=5):
    cache_key = f"recommendations:{user_id}"
    cached_recommendations = redis_client.get(cache_key)
    if cached_recommendations:
        return json.loads(cached_recommendations)

    user_preferences = get_user_preferences(user_id)
    response = index.query(vector=user_preferences, top_k=n, include_metadata=True)
    recommended_products = []
    for match in response["matches"]:
        product_id = match["id"]
        product_doc = await db.collection("products").document(product_id).get()
        if product_doc.exists:
            recommended_products.append(product_doc.to_dict())

    # Cache the recommendations for 1 hour
    redis_client.set(cache_key, json.dumps(recommended_products), ex=3600)
    return recommended_products


async def get_search_products(
    query: str,
    page: int = 1,
    per_page: int = 20,
) -> ProductResults:
    cache_key = f"search:{query}:{page}:{per_page}"
    cached_results = redis_client.get(cache_key)
    if cached_results:
        return ProductResults.parse_raw(cached_results)

    query = query.lower()
    query_embedding = get_text_embedding(query)
    response = index.query(vector=query_embedding, top_k=100, include_metadata=True)

    grouped_products: List[GroupedProduct] = []

    # Group products by meta_sku
    grouped_by_meta_sku = {}
    for match in response["matches"]:
        product_id = match["id"]
        product_doc = await db.collection("products").document(product_id).get()

        if product_doc.exists:
            product_data = product_doc.to_dict()
            product_data["id"] = product_id

            # Convert DatetimeWithNanoseconds to string
            if isinstance(product_data.get("updated_at"), datetime):
                product_data["updated_at"] = product_data["updated_at"].isoformat()
            else:
                product_data["updated_at"] = product_data.get("updated_at")

            # Remove the 'updated_at' field to avoid conflicts
            product_data.pop("updated_at", None)

            meta_sku = product_data.get("meta_sku")
            if meta_sku not in grouped_by_meta_sku:
                grouped_by_meta_sku[meta_sku] = []
            grouped_by_meta_sku[meta_sku].append(Product(**product_data))

    # Sort products within each group, prioritizing non-placeholder images
    for meta_sku, products in grouped_by_meta_sku.items():
        products.sort(
            key=lambda x: (
                x.image_url.endswith("_image_missing.jpg") if x.image_url else True
            )
        )
        grouped_products.append(
            GroupedProduct(
                meta_sku=meta_sku,
                retailer_id=products[0].retailer_id,
                products=products,
            )
        )

    # Sort the final list, prioritizing exact matches and non-placeholder images
    grouped_products.sort(
        key=lambda x: (
            x.products[0].product_name.lower() != query,  # Exact matches first
            (
                x.products[0].image_url.endswith("_image_missing.jpg")
                if x.products[0].image_url
                else True
            ),  # Non-placeholder images next
        )
    )

    # Apply pagination
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_products = grouped_products[start_index:end_index]

    total_results = len(grouped_products)
    total_pages = (total_results + per_page - 1) // per_page

    pagination = Pagination(
        total=total_results,
        count=len(paginated_products),
        per_page=per_page,
        current_page=page,
        total_pages=total_pages,
    )

    result = ProductResults(
        products=paginated_products,
        pagination=pagination,
    )

    # Cache the search results for 15 minutes
    redis_client.set(cache_key, result.json(), ex=900)
    return result
