from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Header,
    Query,
    Request,
    Path,
    status,
)
from typing import List, Optional
from .exceptions import CustomException
from .chat_service import (
    process_chat_message,
    rename_chat,
    get_chat_messages,
    archive_chat,
    delete_chat,
)
from .user_service import get_user_chats
from .auth_service import (
    logout,
    get_firebase_user,
)
import os
from .crud import (
    create_chat_session,
    get_recommended_products,
    search_products,
    create_product,
    get_product,
    update_product,
    delete_product,
    get_user_interactions,
    create_interaction,
    get_dispensaries,
    get_dispensary,
    create_dispensary,
    update_user,
    get_dispensary_inventory,
    create_inventory,
    get_products,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    User,
    UserUpdate,
    Product,
    Dispensary,
    ProductUpdate,
    Inventory,
    Interaction,
    ChatSession,
    ProductCreate,
    InteractionCreate,
    DispensaryCreate,
    InventoryCreate,
)
import httpx
import time
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_firebase_user),
):
    # Access session data
    session = request.session
    chat_id = chat_request.chat_id or session.get("chat_id")

    # Get user_id if authenticated
    user_id = current_user.id if current_user else None
    client_ip = request.client.host
    voice_type = chat_request.voice_type
    session_id = session.get("session_id") or os.urandom(16).hex()
    message = chat_request.message
    user_agent = request.headers.get("User-Agent")
    # Process chat logic
    response = await process_chat_message(
        user_id,
        chat_id,
        session_id,
        client_ip,
        message,
        user_agent,
        voice_type,
        background_tasks,
    )

    # Update session data
    session["chat_id"] = response.chat_id

    return response


@router.get("/user/chats")
async def get_user_chats_endpoint(
    current_user: User = Depends(get_firebase_user),
):
    return await get_user_chats(current_user.id)


@router.get("/chat/messages")
async def get_chat_messages_endpoint(
    chat_id: str = Query(..., description="The chat ID to fetch messages for"),
    current_user: User = Depends(get_firebase_user),
):
    return await get_chat_messages(chat_id)


@router.put("/chat/rename")
async def rename_chat_endpoint(
    chat_id: str = Query(...),
    new_name: str = Query(...),
    authorization: str = Header(None),
):
    return await rename_chat(chat_id, new_name, authorization)


@router.put("/chat/{chat_id}/archive")
async def archive_chat_endpoint(
    chat_id: str = Path(...),
    authorization: str = Header(None),
):
    return await archive_chat(chat_id, authorization)


@router.delete("/chat/{chat_id}")
async def delete_chat_endpoint(
    chat_id: str = Path(...),
    authorization: str = Header(None),
):
    return await delete_chat(chat_id, authorization)


@router.delete("/logout")
async def logout_endpoint(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),
):
    return await logout(fastapi_request, background_tasks, authorization)


@router.put("/users/me", response_model=User)
def update_user_me(
    user: UserUpdate,
    current_user: User = Depends(get_firebase_user),
):
    return update_user(current_user.id, user)


@router.post("/products/", response_model=Product)
def create_product(product: ProductCreate):
    return create_product(product)


@router.get("/products/", response_model=List[Product])
def read_products(skip: int = 0, limit: int = 100):
    products = get_products(skip=skip, limit=limit)
    return products


@router.get("/products/{product_id}", response_model=Product)
def read_product(product_id: str):
    db_product = get_product(product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product


@router.put("/products/{product_id}", response_model=Product)
def update_product(product_id: str, product: ProductUpdate):
    return update_product(product_id, product)


@router.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: str):
    delete_product(product_id)
    return {"ok": True}


@router.post("/interactions/", response_model=Interaction)
def create_interaction(
    interaction: InteractionCreate,
    current_user: User = Depends(get_firebase_user),
):
    return create_interaction(interaction, user_id=current_user.id)


@router.get("/interactions/", response_model=List[Interaction])
def read_interactions(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_firebase_user),
):
    interactions = get_user_interactions(
        user_id=current_user.id, skip=skip, limit=limit
    )
    return interactions


@router.post("/chat/start", response_model=ChatSession)
def start_chat_session(
    current_user: User = Depends(get_firebase_user),
):
    return create_chat_session(user_id=current_user.id)


@router.post("/dispensaries/", response_model=Dispensary)
def create_dispensary(dispensary: DispensaryCreate):
    return create_dispensary(dispensary)


@router.get("/dispensaries/", response_model=List[Dispensary])
def read_dispensaries(skip: int = 0, limit: int = 100):
    dispensaries = get_dispensaries(skip=skip, limit=limit)
    return dispensaries


@router.get("/dispensaries/{dispensary_id}", response_model=Dispensary)
def read_dispensary(dispensary_id: str):
    db_dispensary = get_dispensary(dispensary_id)
    if db_dispensary is None:
        raise HTTPException(status_code=404, detail="Dispensary not found")
    return db_dispensary


@router.post("/inventory/", response_model=Inventory)
def create_inventory(inventory: InventoryCreate):
    return create_inventory(inventory)


@router.get("/inventory/{dispensary_id}", response_model=List[Inventory])
def read_dispensary_inventory(dispensary_id: str):
    inventory = get_dispensary_inventory(dispensary_id)
    return inventory


# Recommendation route
@router.get("/recommendations/", response_model=List[Product])
def get_recommendations(
    current_user: User = Depends(get_firebase_user),
):
    # This is a placeholder. The actual implementation would involve your recommendation algorithm.
    return get_recommended_products(user_id=current_user.id)


# Search route
@router.get("/search/", response_model=List[Product])
def search_products(query: str):
    return search_products(query=query)


@router.get("/live_products")
async def get_live_products(
    current_user: User = Depends(get_firebase_user),
    lat: Optional[float] = Query(None, description="Latitude"),
    lng: Optional[float] = Query(None, description="Longitude"),
    distance: Optional[float] = Query(None, description="Miles"),
    states: List[str] = Query(..., description="List of states, at least one required"),
    retailers: Optional[List[int]] = Query(None, description="List of retailer IDs"),
    brands: Optional[List[int]] = Query(None, description="List of brand IDs"),
    page: int = Query(1, ge=1, description="Page number, starting from 1"),
    skus: Optional[List[str]] = Query(None, description="Cann SKU IDs"),
    brand_name: Optional[str] = Query(None, description="Brand Name"),
    product_name: Optional[str] = Query(None, description="Product Name"),
    display_weight: Optional[str] = Query(None, description="Display Weight"),
    category: Optional[str] = Query(None, description="Category"),
    subcategory: Optional[str] = Query(None, description="Subcategory"),
    tags: Optional[List[str]] = Query(None, description="Product Tags"),
    percentage_thc: Optional[float] = Query(None, description="Percentage THC"),
    percentage_cbd: Optional[float] = Query(None, description="Percentage CBD"),
    mg_thc: Optional[float] = Query(None, description="Mg THC"),
    mg_cbd: Optional[float] = Query(None, description="Mg CBD"),
    quantity_per_package: Optional[int] = Query(
        None, description="Quantity Per Package"
    ),
    medical: Optional[bool] = Query(None, description="Medical"),
    recreational: Optional[bool] = Query(None, description="Recreational"),
    latest_price: Optional[float] = Query(None, description="Latest Price"),
    menu_provider: Optional[str] = Query(None, description="Menu Provider"),
):
    # Construct the query parameters
    params = {
        "lat": lat,
        "lng": lng,
        "distance": distance,
        "states": states,
        "retailers": retailers,
        "brands": brands,
        "page": page,
        "skus": skus,
        "brand_name": brand_name,
        "product_name": product_name,
        "display_weight": display_weight,
        "category": category,
        "subcategory": subcategory,
        "tags": tags,
        "percentage_thc": percentage_thc,
        "percentage_cbd": percentage_cbd,
        "mg_thc": mg_thc,
        "mg_cbd": mg_cbd,
        "quantity_per_package": quantity_per_package,
        "medical": medical,
        "recreational": recreational,
        "latest_price": latest_price,
        "menu_provider": menu_provider,
    }

    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    # Make the API call
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('CANNMENUS_API_KEY')}",
                "Content-Type": "application/json",
            }
            response = await client.get(
                "https://api.cannmenus.com/v1/products", params=params, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/scrape-retailer-products/{retailer_id}")
async def scrape_retailer_products(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_firebase_user),
    retailer_id: int = Path(..., description="The ID of the retailer to scrape"),
):
    logger.info(f"Starting product scrape for retailer_id: {retailer_id}")
    try:
        # Authenticate user
        user_id = current_user.id
        logger.info(f"User authenticated with user_id: {user_id}")

        # Start the scraping process in the background
        background_tasks.add_task(scrape_and_store_products, retailer_id, user_id)

        return {
            "message": "Product scraping started",
            "retailer_id": retailer_id,
            "user_id": user_id,
        }

    except HTTPException as http_error:
        logger.error(f"HTTP error occurred: {http_error}")
        raise http_error
    except Exception as e:
        logger.error(f"Error occurred while initiating product scrape: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


def scrape_and_store_products(retailer_id: int, user_id: str):
    logger.info(f"Scraping products for retailer_id: {retailer_id}")
    try:
        # Initialize variables for pagination
        page = 1
        all_products = []
        total_pages = 1
        requests_count = 0
        start_time = time.time()

        headers = {
            "X-Token": f"{os.getenv('CANNMENUS_API_KEY')}",
        }
        while page <= total_pages:
            # Implement rate limiting
            if requests_count >= 10:  # Limit to 10 requests per minute
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    logger.info(
                        f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds"
                    )
                    time.sleep(sleep_time)
                requests_count = 0
                start_time = time.time()

            params = {"retailers": [retailer_id], "page": page, "states": "michigan"}
            try:
                with httpx.Client() as client:
                    response = client.get(
                        "https://api.cannmenus.com/v1/products",
                        params=params,
                        headers=headers,
                        timeout=10.0,
                    )
                response.raise_for_status()
                data = response.json()

                requests_count += 1

                products = data.get("data", [])
                if not products:
                    break  # No more products to fetch

                # Flatten and process the products
                for product_group in products:
                    sku = product_group.get("sku")
                    for product in product_group.get("products", []):
                        flattened_product = {
                            "retailer_id": retailer_id,
                            "sku": sku,
                            **product,
                        }
                        all_products.append(flattened_product)

                total_pages = data.get("pagination", {}).get("total_pages", total_pages)
                logger.info(
                    f"Fetched page {page} of {total_pages} for retailer {retailer_id}"
                )
                page += 1

                # Add a small delay between requests
                time.sleep(1)

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
                )
                break
            except httpx.RequestError as e:
                logger.error(f"An error occurred while requesting: {e}")
                break

        # Process and store the products in Firestore
        products_ref = db.collection("products")

        # Update products
        batch = db.batch()
        updated_product_ids = set()
        count = 0
        for product in all_products:
            product_id = product.get("sku")
            if not product_id:
                logger.warning(f"Skipping product without SKU: {product}")
                continue
            product_ref = products_ref.document(product_id)
            batch.set(
                product_ref,
                {
                    **product,
                    "last_updated": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            updated_product_ids.add(product_id)
            count += 1

            # If we've reached the batch limit, commit and reset
            if count % 500 == 0:
                batch.commit()
                batch = db.batch()

        # Commit any remaining updates
        if count % 500 != 0:
            batch.commit()

        # Delete products that are no longer associated with this retailer
        query = products_ref.where("retailer_id", "==", retailer_id)
        existing_products = query.stream()

        delete_batch = db.batch()
        delete_count = 0
        for doc in existing_products:
            if doc.id not in updated_product_ids:
                delete_batch.delete(doc.reference)
                delete_count += 1

            # If we've reached the batch limit, commit and reset
            if delete_count % 500 == 0:
                delete_batch.commit()
                delete_batch = db.batch()

        # Commit any remaining deletes
        if delete_count % 500 != 0:
            delete_batch.commit()

        logger.info(
            f"Successfully scraped and stored {len(updated_product_ids)} products for retailer_id: {retailer_id}"
        )

    except Exception as e:
        logger.error(f"Error occurred while scraping products: {e}")
