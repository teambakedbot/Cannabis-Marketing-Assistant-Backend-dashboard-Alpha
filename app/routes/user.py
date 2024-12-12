from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Header,
    Query,
    Request,
)
from typing import Any, Dict, List
from ..services.auth_service import logout, get_firebase_user
from ..crud.crud import (
    create_product,
    get_user_interactions,
    create_interaction,
    update_user,
    get_user_theme,
    save_user_theme,
)
from ..models.schemas import (
    User,
    UserUpdate,
    Product,
    Interaction,
    ProductCreate,
    InteractionCreate,
)
from ..config.config import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


async def handle_exception(e: Exception) -> None:
    """Helper function to log errors."""
    logger.error(f"Error occurred: {str(e)}")
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
        await handle_exception(e)


@router.put("/users/me", response_model=User)
async def update_user_me(
    user: UserUpdate,
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await update_user(current_user.id, user)
    except Exception as e:
        await handle_exception(e)


@router.post("/products/", response_model=Product)
async def create_product_endpoint(product: ProductCreate):
    try:
        return await create_product(product)
    except Exception as e:
        await handle_exception(e)


@router.get("/users/theme", response_model=Dict[str, Any])
async def get_user_theme_endpoint(current_user: User = Depends(get_firebase_user)):
    try:
        return await get_user_theme(current_user.id)
    except Exception as e:
        await handle_exception(e)


@router.post("/users/theme", response_model=Dict[str, Any])
async def save_user_theme_endpoint(
    theme: Dict[str, Any], current_user: User = Depends(get_firebase_user)
):
    try:
        return await save_user_theme(current_user.id, theme)
    except Exception as e:
        await handle_exception(e)


@router.post("/interactions/", response_model=Interaction)
async def create_interaction_endpoint(
    interaction: InteractionCreate,
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await create_interaction(interaction, user_id=current_user.id)
    except Exception as e:
        await handle_exception(e)


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
        await handle_exception(e)
