from ..utils.firebase_utils import db
from ..models import schemas
from datetime import datetime
from typing import Any, Dict, List, Optional
from passlib.context import CryptContext
from fastapi import HTTPException
import uuid
from google.cloud import firestore
from redis import Redis
import json
from functools import lru_cache
from firebase_admin import firestore
from google.cloud.firestore_v1.async_client import AsyncClient
from ..config.config import settings
import os
from google.cloud.firestore_v1.base_query import FieldFilter
from ..models.schemas import Product
from itertools import groupby
from operator import itemgetter

# Initialize Redis client
redis_url = settings.REDISCLOUD_URL
redis_client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)


@lru_cache(maxsize=100)
async def get_user_by_email(email: str):
    """
    Get a user by their email.
    """
    users_ref = db.collection("users")
    query = users_ref.where("email", "==", email).limit(1)
    results = await query.get()
    for doc in results:
        user_data = doc.to_dict()
        user_data["id"] = doc.id
        return schemas.User(**user_data)
    return None


async def create_user(user: schemas.UserCreate):
    """
    Create a new user with a hashed password.
    """
    hashed_password = user.password
    user_data = user.dict()
    user_data["hashed_password"] = hashed_password
    user_data["created_at"] = firestore.SERVER_TIMESTAMP
    user_data["updated_at"] = firestore.SERVER_TIMESTAMP
    user_data["is_active"] = True
    user_data["is_superuser"] = False
    user_id = str(uuid.uuid4())
    user_ref = db.collection("users").document(user_id)
    await user_ref.set(user_data)
    user_data["id"] = user_id
    return schemas.User(**user_data)


@lru_cache(maxsize=1000)
async def get_user(user_id: str):
    user_ref = db.collection("users").document(user_id)
    doc = await user_ref.get()
    if doc.exists:
        user_data = doc.to_dict()
        user_data["id"] = doc.id
        return schemas.User(**user_data)
    else:
        return None


async def update_user(user_id: str, user: schemas.UserUpdate):
    user_ref = db.collection("users").document(user_id)
    if not await user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")
    update_data = user.dict(exclude_unset=True)
    update_data["updated_at"] = firestore.SERVER_TIMESTAMP
    await user_ref.update(update_data)
    updated_user = await user_ref.get().to_dict()
    updated_user["id"] = user_id
    return schemas.User(**updated_user)


def get_default_theme() -> Dict[str, str]:
    return {
        "defaultLanguage": "English",
        "defaultTheme": "Light",
        "botVoice": "Male",
        "allowedSites": ["*"],
        "colors": {
            "primaryColor": "#22AD85",
            "secondaryColor": "#23504A",
            "backgroundColor": "#1E1E1E",
            "headerColor": "#2C2C2C",
            "textColor": "#FFFFFF",
            "textSecondaryColor": "#FFFFFF",
        },
    }


async def save_user_theme(user_id: str, theme: Dict[str, str]):
    """
    Save or update the user's theme preferences in the themes table.
    If the theme doesn't exist, it creates a new one.
    """
    theme_ref = db.collection("themes").document(user_id)
    theme_doc = await theme_ref.get()
    redis_client.delete(f"user_theme:{user_id}")

    if not theme_doc.exists:
        await theme_ref.set(theme)
        return {"message": "New theme created successfully"}
    else:
        await theme_ref.update(theme)
        return {"message": "Theme updated successfully"}


async def get_user_theme(user_id: str) -> Dict[str, Any]:
    """
    Retrieve the user's theme preferences from the themes table.
    If no theme exists, create a default one and return it.
    """
    print(f"Getting user theme for user_id: {user_id}")
    cache_key = f"user_theme:{user_id}"
    cached_theme = redis_client.get(cache_key)
    if cached_theme:
        return json.loads(cached_theme)

    theme_ref = db.collection("themes").document(user_id)
    theme_doc = await theme_ref.get()

    if not theme_doc.exists:
        default_theme = get_default_theme()
        await theme_ref.set(default_theme)
        redis_client.set(
            cache_key, json.dumps(default_theme), ex=3600
        )  # Cache for 1 hour
        return default_theme

    theme = theme_doc.to_dict()
    redis_client.set(cache_key, json.dumps(theme), ex=3600)  # Cache for 1 hour
    return theme


async def create_product(product: schemas.ProductCreate):
    product_data = product.dict()
    product_data["created_at"] = firestore.SERVER_TIMESTAMP
    product_data["updated_at"] = firestore.SERVER_TIMESTAMP
    product_id = str(uuid.uuid4())
    product_ref = db.collection("products").document(product_id)
    await product_ref.set(product_data)
    product_data["id"] = product_id
    return schemas.Product(**product_data)


async def get_products(
    skip: int = 0,
    limit: int = 100,
    retailers: Optional[List[int]] = None,
    product_name: Optional[str] = None,
    states: Optional[List[str]] = None,
):
    products_ref = db.collection("products")
    query = products_ref

    if retailers:
        query = query.where(filter=FieldFilter("retailer_id", "in", retailers))
    if product_name:
        query = query.where(filter=FieldFilter("product_name", "==", product_name))

    # Get total count
    total_count = len(list(await query.get()))

    # Apply pagination
    query = query.offset(skip).limit(limit)
    docs = await query.get()

    # Group products by meta_sku
    products_list = []
    for doc in docs:
        product_data = doc.to_dict()
        product_data["id"] = doc.id
        products_list.append(product_data)

    # Sort the list by meta_sku to prepare for grouping
    products_list.sort(key=itemgetter("meta_sku"))

    # Group products by meta_sku
    grouped_products = []
    for meta_sku, group in groupby(products_list, key=itemgetter("meta_sku")):
        group_list = list(group)

        # Sort products within the group, prioritizing non-placeholder images
        group_list.sort(
            key=lambda x: x.get("image_url", "").endswith("_image_missing.jpg")
        )

        grouped_product = {
            "meta_sku": meta_sku,
            "retailer_id": group_list[0]["retailer_id"],
            "products": group_list,
        }
        grouped_products.append(grouped_product)

    pagination = {
        "total": total_count,
        "count": len(grouped_products),
        "per_page": limit,
        "current_page": skip // limit + 1,
        "total_pages": -(-total_count // limit),  # Ceiling division
    }

    return {"products": grouped_products, "pagination": pagination}


@lru_cache(maxsize=1000)
async def get_product(product_id: str):
    product_ref = await db.collection("products").document(product_id).get()
    if product_ref.exists:
        product_data = product_ref.to_dict()
        product_data["id"] = product_ref.id
        return schemas.Product(**product_data)
    else:
        return None


async def update_product(product_id: str, product: schemas.ProductUpdate):
    product_ref = db.collection("products").document(product_id)
    if not await product_ref.get().exists:
        raise HTTPException(status_code=404, detail="Product not found")
    update_data = product.dict(exclude_unset=True)
    update_data["updated_at"] = firestore.SERVER_TIMESTAMP
    await product_ref.update(update_data)
    updated_product = await product_ref.get().to_dict()
    updated_product["id"] = product_id
    return schemas.Product(**updated_product)


async def delete_product(product_id: str):
    product_ref = db.collection("products").document(product_id)
    if not await product_ref.get().exists:
        raise HTTPException(status_code=404, detail="Product not found")
    await product_ref.delete()
    return {"message": "Product deleted successfully"}


async def create_interaction(interaction: schemas.InteractionCreate, user_id: str):
    interaction_data = interaction.dict()
    interaction_data["user_id"] = user_id
    interaction_data["timestamp"] = datetime.utcnow()
    interaction_id = str(uuid.uuid4())
    interaction_ref = db.collection("interactions").document(interaction_id)
    await interaction_ref.set(interaction_data)
    interaction_data["id"] = interaction_id
    return schemas.Interaction(**interaction_data)


async def get_user_interactions(user_id: str, skip: int = 0, limit: int = 100):
    interactions_ref = db.collection("interactions")
    query = interactions_ref.where("user_id", "==", user_id).offset(skip).limit(limit)
    docs = await query.get()
    interactions = []
    for doc in docs:
        interaction_data = doc.to_dict()
        interaction_data["id"] = doc.id
        interactions.append(schemas.Interaction(**interaction_data))
    return interactions


async def create_chat_session(user_id: str):
    chat_session_data = {
        "user_id": user_id,
        "start_time": datetime.utcnow(),
        "end_time": None,
        "session_data": None,
    }
    session_id = str(uuid.uuid4())
    chat_session_ref = db.collection("chat_sessions").document(session_id)
    await chat_session_ref.set(chat_session_data)
    chat_session_data["id"] = session_id
    return schemas.ChatSession(**chat_session_data)


async def create_chat_message(session_id: str, message: schemas.ChatMessageCreate):
    message_data = message.dict()
    message_data["timestamp"] = datetime.utcnow()
    message_id = str(uuid.uuid4())
    chat_message_ref = (
        db.collection("chat_sessions")
        .document(session_id)
        .collection("messages")
        .document(message_id)
    )
    await chat_message_ref.set(message_data)
    message_data["id"] = message_id
    return schemas.ChatMessage(**message_data)


async def get_chat_messages(session_id: str):
    messages_ref = (
        db.collection("chat_sessions").document(session_id).collection("messages")
    )
    docs = await messages_ref.order_by("timestamp").get()
    messages = []
    for doc in docs:
        message_data = doc.to_dict()
        message_data["id"] = doc.id
        messages.append(schemas.ChatMessage(**message_data))
    return messages


async def create_dispensary(dispensary: schemas.DispensaryCreate):
    dispensary_data = dispensary.dict()
    dispensary_data["created_at"] = datetime.utcnow()
    dispensary_data["updated_at"] = datetime.utcnow()
    retailer_id = str(uuid.uuid4())
    dispensary_ref = db.collection("dispensaries").document(retailer_id)
    await dispensary_ref.set(dispensary_data)
    dispensary_data["id"] = retailer_id
    return schemas.Dispensary(**dispensary_data)


async def get_dispensaries(skip: int = 0, limit: int = 100):
    dispensaries_ref = db.collection("dispensaries")
    docs = await dispensaries_ref.offset(skip).limit(limit).get()
    dispensaries = []
    for doc in docs:
        dispensary_data = doc.to_dict()
        dispensary_data["id"] = doc.id
        dispensaries.append(schemas.Dispensary(**dispensary_data))
    return dispensaries


@lru_cache(maxsize=1000)
async def get_dispensary(retailer_id: str):
    dispensary_ref = db.collection("dispensaries").document(retailer_id)
    doc = await dispensary_ref.get()
    if doc.exists:
        dispensary_data = doc.to_dict()
        dispensary_data["id"] = doc.id
        return schemas.Dispensary(**dispensary_data)
    else:
        return None


async def create_inventory(inventory: schemas.InventoryCreate):
    inventory_data = inventory.dict()
    inventory_data["created_at"] = firestore.SERVER_TIMESTAMP
    inventory_data["updated_at"] = firestore.SERVER_TIMESTAMP
    inventory_id = str(uuid.uuid4())
    inventory_ref = db.collection("inventory").document(inventory_id)
    await inventory_ref.set(inventory_data)
    inventory_data["id"] = inventory_id
    return schemas.Inventory(**inventory_data)


async def get_dispensary_inventory(retailer_id: str):
    inventory_ref = db.collection("inventory")
    query = inventory_ref.where("retailer_id", "==", retailer_id)
    docs = await query.get()
    inventory = []
    for doc in docs:
        inventory_data = doc.to_dict()
        inventory_data["id"] = doc.id
        inventory.append(schemas.Inventory(**inventory_data))
    return inventory


@lru_cache(maxsize=100)
async def get_recommended_products(user_id: str) -> List[schemas.Product]:
    products_ref = db.collection("products")
    docs = (
        await products_ref.order_by("updated_at", direction=firestore.Query.DESCENDING)
        .limit(5)
        .get()
    )
    products = []
    for doc in docs:
        product_data = doc.to_dict()
        product_data["id"] = doc.id
        products.append(schemas.Product(**product_data))
    return products


async def search_products(query: str) -> List[schemas.Product]:
    products_ref = db.collection("products")
    # Firestore does not support case-insensitive searches directly
    # This is a workaround by fetching all products and filtering in code
    docs = await products_ref.get()
    products = []
    for doc in docs:
        product_data = doc.to_dict()
        if query.lower() in product_data.get("name", "").lower():
            product_data["id"] = doc.id
            products.append(schemas.Product(**product_data))
    return products


async def create_order(order: schemas.OrderRequest):
    order_data = order.dict()
    order_data["created_at"] = datetime.utcnow()
    order_data["updated_at"] = datetime.utcnow()
    order_data["status"] = "pending"
    order_id = str(uuid.uuid4())
    order_ref = db.collection("orders").document(order_id)
    await order_ref.set(order_data)
    order_data["id"] = order_id
    return schemas.Order(**order_data)
