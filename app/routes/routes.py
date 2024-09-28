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
from ..services.recommendation_system import get_search_products
from typing import Any, Dict, List, Optional
from ..exceptions.exceptions import CustomException
from ..services.chat_service import (
    process_chat_message,
    rename_chat,
    get_chat_messages,
    archive_chat,
    delete_chat,
    record_feedback,
    retry_message,
)
from ..services.user_service import get_user_chats
from ..services.auth_service import (
    logout,
    get_firebase_user,
)
import os
from ..crud.crud import (
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
    get_user_theme,
    save_user_theme,
    create_order,
)
from ..models.schemas import (
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
    FeedbackCreate,
    MessageRetry,
    ProductCreate,
    InteractionCreate,
    DispensaryCreate,
    InventoryCreate,
    BaseModel,
    ContactInfo,
    OrderRequest,
    ProductResults,
    ChatResponse,
)
import httpx
import time
import logging
from ..utils.firebase_utils import db, firestore
from ..utils.redis_config import get_redis, FirestoreEncoder
from redis.asyncio import Redis
import json
from ..config.config import settings, logger
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client
from dotenv import load_dotenv


# from fastapi.middleware.throttle import ThrottleMiddleware

# Load environment variables
load_dotenv()

router = APIRouter()

# Add rate limiting middleware
# router.add_middleware(ThrottleMiddleware, rate_limit="100/minute")


@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    redis: Redis = Depends(get_redis),
    current_user: Optional[User] = Depends(get_firebase_user),
):
    try:
        session = request.session
        chat_id = chat_request.chat_id or session.get("chat_id")
        user_id = current_user.id if current_user else None
        client_ip = request.client.host
        voice_type = chat_request.voice_type
        session_id = session.get("session_id") or os.urandom(16).hex()
        message = chat_request.message
        user_agent = request.headers.get("User-Agent")

        response = await process_chat_message(
            user_id=user_id,
            chat_id=chat_id,
            session_id=session_id,
            client_ip=client_ip,
            message=message,
            user_agent=user_agent,
            voice_type=voice_type,
            background_tasks=background_tasks,
            redis_client=redis,
        )
        return response
    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user/chats")
async def get_user_chats_endpoint(
    current_user: User = Depends(get_firebase_user),
    redis: Redis = Depends(get_redis),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        return await get_user_chats(current_user.id, redis, page, page_size)
    except Exception as e:
        logger.error(f"Error in get_user_chats_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/chat/messages")
async def get_chat_messages_endpoint(
    chat_id: str = Query(..., description="The chat ID to fetch messages for"),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await get_chat_messages(chat_id)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in get_chat_messages_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/chat/rename")
async def rename_chat_endpoint(
    chat_id: str = Query(...),
    new_name: str = Query(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await rename_chat(chat_id, new_name, current_user.id)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in rename_chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/chat/{chat_id}/archive")
async def archive_chat_endpoint(
    chat_id: str = Path(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await archive_chat(chat_id, current_user.id)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in archive_chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/chat/{chat_id}")
async def delete_chat_endpoint(
    chat_id: str = Path(...),
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await delete_chat(chat_id, current_user.id)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in delete_chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/logout")
async def logout_endpoint(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None),
):
    try:
        return await logout(fastapi_request, background_tasks, authorization)
    except Exception as e:
        logger.error(f"Error in logout_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/users/me", response_model=User)
def update_user_me(
    user: UserUpdate,
    current_user: User = Depends(get_firebase_user),
):
    try:
        return update_user(current_user.id, user)
    except Exception as e:
        logger.error(f"Error in update_user_me: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/products/", response_model=Product)
def create_product_endpoint(product: ProductCreate):
    try:
        return create_product(product)
    except Exception as e:
        logger.error(f"Error in create_product_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/users/theme", response_model=Dict[str, Any])
async def get_user_theme_endpoint(current_user: User = Depends(get_firebase_user)):
    try:
        return await get_user_theme(current_user.id)
    except Exception as e:
        logger.error(f"Error in get_user_theme_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/users/theme", response_model=Dict[str, Any])
async def save_user_theme_endpoint(
    theme: Dict[str, Any], current_user: User = Depends(get_firebase_user)
):
    try:
        return await save_user_theme(current_user.id, theme)
    except Exception as e:
        logger.error(f"Error in save_user_theme_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/products/", response_model=ProductResults)
async def read_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    redis: Redis = Depends(get_redis),
    retailers: Optional[List[int]] = Query(None, description="List of retailer IDs"),
):
    try:
        skip = (page - 1) * limit
        cache_key = f"products:{skip}:{limit}:{','.join(map(str, retailers or []))}"
        cached_products = await redis.get(cache_key)
        if cached_products:
            return json.loads(cached_products)

        results = await get_products(skip=skip, limit=limit, retailers=retailers)
        await redis.set(
            cache_key, json.dumps(results, cls=FirestoreEncoder), ex=3600
        )  # Cache for 1 hour
        return results
    except Exception as e:
        logger.error(f"Error in read_products: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/products/search", response_model=ProductResults)
async def read_search_products(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    try:
        products = await get_search_products(query, page, per_page)
        return products
    except Exception as e:
        logger.error(f"Error in read_search_products: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/products/{product_id}", response_model=Product)
async def read_product(product_id: str, redis: Redis = Depends(get_redis)):
    try:
        cached_product = await redis.get(f"product:{product_id}")
        if cached_product:
            return json.loads(cached_product)
        db_product = get_product(product_id)
        if db_product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        await redis.set(
            f"product:{product_id}",
            json.dumps(db_product, cls=FirestoreEncoder),
            ex=3600,
        )
        return db_product
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in read_product: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/products/{product_id}", response_model=Product)
async def update_product_endpoint(product_id: str, product: ProductUpdate):
    try:
        return await update_product(product_id, product)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in update_product_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product_endpoint(product_id: str):
    try:
        await delete_product(product_id)
        return {"ok": True}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in delete_product_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interactions/", response_model=Interaction)
async def create_interaction_endpoint(
    interaction: InteractionCreate,
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await create_interaction(interaction, user_id=current_user.id)
    except Exception as e:
        logger.error(f"Error in create_interaction_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/interactions/", response_model=List[Interaction])
async def read_interactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_firebase_user),
):
    try:
        interactions = await get_user_interactions(
            user_id=current_user.id, skip=skip, limit=limit
        )
        return interactions
    except Exception as e:
        logger.error(f"Error in read_interactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat/start", response_model=ChatSession)
async def start_chat_session(
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await create_chat_session(user_id=current_user.id)
    except Exception as e:
        logger.error(f"Error in start_chat_session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dispensaries/", response_model=Dispensary)
async def create_dispensary_endpoint(dispensary: DispensaryCreate):
    try:
        return await create_dispensary(dispensary)
    except Exception as e:
        logger.error(f"Error in create_dispensary_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dispensaries/", response_model=List[Dispensary])
async def read_dispensaries(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    try:
        dispensaries = await get_dispensaries(skip=skip, limit=limit)
        return dispensaries
    except Exception as e:
        logger.error(f"Error in read_dispensaries: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dispensaries/{retailer_id}", response_model=Dispensary)
async def read_dispensary(retailer_id: str):
    try:
        db_dispensary = get_dispensary(retailer_id)
        if db_dispensary is None:
            raise HTTPException(status_code=404, detail="Dispensary not found")
        return db_dispensary
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in read_dispensary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/inventory/", response_model=Inventory)
async def create_inventory_endpoint(inventory: InventoryCreate):
    try:
        return await create_inventory(inventory)
    except Exception as e:
        logger.error(f"Error in create_inventory_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/inventory/{retailer_id}", response_model=List[Inventory])
async def read_dispensary_inventory_endpoint(retailer_id: str):
    try:
        inventory = await get_dispensary_inventory(retailer_id)
        return inventory
    except Exception as e:
        logger.error(f"Error in read_dispensary_inventory_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations/", response_model=List[Product])
async def get_recommendations_endpoint(
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await get_recommended_products(user_id=current_user.id)
    except Exception as e:
        logger.error(f"Error in get_recommendations_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/search/", response_model=List[Product])
async def search_products_endpoint(query: str = Query(..., min_length=1)):
    try:
        return await search_products(query=query)
    except Exception as e:
        logger.error(f"Error in search_products_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/live_products")
async def get_live_products(
    current_user: User = Depends(get_firebase_user),
    redis: Redis = Depends(get_redis),
):
    try:
        params = {
            k: v
            for k, v in locals().items()
            if v is not None and k not in ["current_user", "redis"]
        }
        cache_key = f"live_products:{hash(frozenset(params.items()))}"
        cached_result = await redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        async with httpx.AsyncClient() as client:
            headers = {
                "X-Token": f"{settings.CANNMENUS_API_KEY}",
                "Content-Type": "application/json",
            }
            response = await client.get(
                "https://api.cannmenus.com/v1/products", params=params, headers=headers
            )
            response.raise_for_status()
            result = response.json()
            await redis.set(
                cache_key, json.dumps(result), ex=300
            )  # Cache for 5 minutes
            return result
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_live_products: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/scrape-retailer-products/{retailer_id}")
async def scrape_retailer_products(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_firebase_user),
    retailer_id: int = Path(..., description="The ID of the retailer to scrape"),
):
    try:
        logger.info(f"Starting product scrape for retailer_id: {retailer_id}")
        user_id = current_user.id
        logger.info(f"User authenticated with user_id: {user_id}")
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
            "X-Token": f"{settings.CANNMENUS_API_KEY}",
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


@router.post("/scrape-retailers")
async def scrape_retailers(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_firebase_user),
):
    logger.info("Starting retailer scrape")
    try:
        # Authenticate user
        user_id = current_user.id
        logger.info(f"User authenticated with user_id: {user_id}")

        # Start the scraping process in the background
        background_tasks.add_task(scrape_and_store_retailers, user_id)

        return {
            "message": "Retailer scraping started",
            "user_id": user_id,
        }

    except HTTPException as http_error:
        logger.error(f"HTTP error occurred: {http_error}")
        raise http_error
    except Exception as e:
        logger.error(f"Error occurred while initiating retailer scrape: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


def scrape_and_store_retailers(user_id: str):
    logger.info("Scraping retailers")
    try:
        # Initialize variables for pagination
        page = 1
        all_retailers = []
        total_pages = 1
        requests_count = 0
        start_time = time.time()

        headers = {
            "X-Token": f"{settings.CANNMENUS_API_KEY}",
            "Content-Type": "application/json",
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

            params = {"page": page, "city": "Detroit"}
            try:
                with httpx.Client() as client:
                    response = client.get(
                        "https://api.cannmenus.com/v1/retailers",
                        params=params,
                        headers=headers,
                        timeout=10.0,
                    )
                response.raise_for_status()
                data = response.json()
                requests_count += 1

                retailers = data.get("data", [])
                if not retailers:
                    break  # No more retailers to fetch

                all_retailers.extend(retailers)

                total_pages = data.get("pagination", {}).get("total_pages", total_pages)
                logger.info(f"Fetched page {page} of {total_pages}")
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

        # Process and store the retailers in Firestore
        retailers_ref = db.collection("retailers")

        # Update retailers
        batch = db.batch()
        updated_retailer_ids = set()
        count = 0

        for retailer in all_retailers:
            retailer_id = str(retailer.get("id"))  # Convert id to string
            if not retailer_id:
                logger.warning(f"Skipping retailer without retailer_id: {retailer}")
                continue
            retailer_ref = retailers_ref.document(retailer_id)
            batch.set(
                retailer_ref,
                {
                    **retailer,
                    "last_updated": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            updated_retailer_ids.add(retailer_id)
            count += 1

            # If we've reached the batch limit, commit and reset
            if count % 500 == 0:
                batch.commit()
                batch = db.batch()

        # Commit any remaining updates
        if count % 500 != 0:
            batch.commit()

        logger.info(
            f"Successfully scraped and stored {len(updated_retailer_ids)} retailers"
        )

    except Exception as e:
        logger.error(f"Error occurred while scraping retailers: {e}")


@router.post("/feedback")
async def record_feedback_endpoint(
    feedback: FeedbackCreate,
    current_user: User = Depends(get_firebase_user),
):
    """
    Record user feedback for a specific message.
    """
    try:
        result = await record_feedback(
            user_id=current_user.id,
            message_id=feedback.message_id,
            feedback_type=feedback.feedback_type,
        )
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retry")
async def retry_message_endpoint(
    retry: MessageRetry,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_firebase_user),
    redis: Redis = Depends(get_redis),
):
    """
    Retry a specific message in the chat history.
    """
    try:
        result = await retry_message(
            user_id=current_user.id,
            message_id=retry.message_id,
            background_tasks=background_tasks,
            redis=redis,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def send_sms(to_phone: str, body: str):
    try:
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        message = client.messages.create(
            body=body, from_=os.getenv("TWILIO_PHONE_NUMBER"), to=to_phone
        )
        print(f"SMS sent. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False


def send_email(to_email: str, subject: str, body: str):
    message = Mail(
        from_email=os.getenv("SENDGRID_FROM_EMAIL"),
        to_emails=to_email,
        subject=subject,
        html_content=body,
    )
    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"Email sent to {to_email}. Status Code: {response.status_code}")
        return True
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
        return False


@router.post("/checkout")
async def place_order(
    order: OrderRequest,
    current_user: User = Depends(get_firebase_user),
):
    try:
        # Create order in the database
        new_order = await create_order(order)

        # Prepare customer email content
        customer_subject = "Order Confirmation"
        customer_body = f"""
        <html>
        <body>
        <h2>Dear {order.name},</h2>

        <p>Thank you for your order. We have received the following details:</p>

        <ul>
        <li><strong>Order ID:</strong> {new_order.id}</li>
        <li><strong>Name:</strong> {order.name}</li>
        <li><strong>Email:</strong> {order.contact_info.email}</li>
        <li><strong>Phone:</strong> {order.contact_info.phone or 'Not provided'}</li>
        </ul>

        <h3>Order Details:</h3>
        <pre>{order.cart}</pre>

        <p>We will contact you soon with pickup details.</p>

        <p>Best regards,<br>Your Store Team</p>
        </body>
        </html>
        """

        # Prepare retailer email content
        retailer_subject = "New Order Received"
        retailer_body = f"""
        <html>
        <body>
        <h2>New Order Received</h2>

        <p>A new order has been placed with the following details:</p>

        <ul>
        <li><strong>Order ID:</strong> {new_order.id}</li>
        <li><strong>Customer Name:</strong> {order.name}</li>
        <li><strong>Email:</strong> {order.contact_info.email}</li>
        <li><strong>Phone:</strong> {order.contact_info.phone or 'Not provided'}</li>
        </ul>

        <h3>Order Details:</h3>
        <pre>{order.cart}</pre>

        <p>Please process this order as soon as possible.</p>
        </body>
        </html>
        """

        # Send emails
        if order.contact_info.phone is not None:
            customer_sms_sent = send_sms(
                order.contact_info.phone, f"New order received: {new_order.id}"
            )

        if order.contact_info.email is not None:
            customer_email_sent = send_email(
                order.contact_info.email, customer_subject, customer_body
            )

        retailer_email_sent = send_email(
            os.getenv("RETAILER_EMAIL"), retailer_subject, retailer_body
        )

        if (customer_sms_sent or customer_email_sent) and retailer_email_sent:
            return {
                "message": "Order placed successfully and confirmation emails sent."
            }
        elif customer_sms_sent or customer_email_sent:
            return {
                "message": "Order placed successfully and customer email sent, but there was an issue sending the retailer email."
            }
        elif retailer_email_sent:
            return {
                "message": "Order placed successfully and retailer email sent, but there was an issue sending the customer email."
            }
        else:
            return {
                "message": "Order placed successfully, but there were issues sending confirmation emails."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
