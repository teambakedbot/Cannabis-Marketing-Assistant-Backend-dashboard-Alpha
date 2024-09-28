import numpy as np
from ..config.config import settings
from ..utils.firebase_utils import db
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from ..models.schemas import ChatResponse, Product, Pagination
from redis import Redis
import json
from functools import lru_cache
import os
from typing import List
from datetime import datetime
import uuid
from ..pinecone.data_ingestion import fetch_and_upsert_products

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("product-index")

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Initialize Redis
redis_url = os.getenv("REDISCLOUD_URL", "redis://localhost:6379")
redis_client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)


@lru_cache(maxsize=1000)
def get_product_embedding(product_data):
    text = f"{product_data.get('name', '')} {product_data.get('category', '')} {product_data.get('description', '')}"
    embedding = embed_model.embed(text)
    return embedding


async def update_embeddings():
    await fetch_and_upsert_products()


@lru_cache(maxsize=1000)
def get_user_preferences(user_id):
    try:
        interactions = (
            db.collection("interactions").where("user_id", "==", user_id).get()
        )
        preferences = np.zeros(100)  # Assume 100-dimensional embeddings
        for interaction in interactions:
            product = db.collection("products").document(interaction.product_id).get()
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
        product_doc = db.collection("products").document(product_id).get()
        if product_doc.exists:
            recommended_products.append(product_doc.to_dict())

    # Cache the recommendations for 1 hour
    redis_client.set(cache_key, json.dumps(recommended_products), ex=3600)
    return recommended_products


async def get_search_products(
    query: str, chat_id: str = uuid.uuid4(), page: int = 1, per_page: int = 5
) -> ChatResponse:
    cache_key = f"search:{query}:{page}:{per_page}"
    cached_results = redis_client.get(cache_key)
    if cached_results:
        return ChatResponse.parse_raw(cached_results)

    query = query.lower()
    query_embedding = embed_model.get_text_embedding(query)
    response = index.query(vector=query_embedding, top_k=100, include_metadata=True)

    search_products: List[Product] = []
    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    for match in response["matches"][start_index:end_index]:
        product_id = match["id"]
        product_doc = db.collection("products").document(product_id).get()

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

            search_products.append(Product(**product_data))

    total_results = len(response["matches"])
    total_pages = (total_results + per_page - 1) // per_page

    pagination = Pagination(
        total=total_results,
        count=len(search_products),
        per_page=per_page,
        current_page=page,
        total_pages=total_pages,
    )

    result = ChatResponse(
        response=f"Here are the search results for '{query}'",
        data={"products": search_products},
        pagination=pagination,
        chat_id=chat_id,
    )

    # Cache the search results for 15 minutes
    redis_client.set(cache_key, result.json(), ex=900)
    return result
