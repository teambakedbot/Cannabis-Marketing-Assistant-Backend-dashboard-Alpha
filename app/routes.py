from .auth import (
    create_access_token,
)
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


router = APIRouter()


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
