from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)
from typing import List
from ..crud.crud import (
    get_dispensaries,
    get_dispensary,
    create_dispensary,
)
from ..models.schemas import (
    Dispensary,
    DispensaryCreate,
)
from ..config.config import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["dispensary"],
    responses={404: {"description": "Not found"}},
)


async def handle_exception(e: Exception) -> HTTPException:
    """Helper function to handle exceptions and log errors."""
    logger.error(f"Error: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dispensaries/", response_model=Dispensary)
async def create_dispensary_endpoint(dispensary: DispensaryCreate):
    """Create a new dispensary."""
    try:
        return await create_dispensary(dispensary)
    except Exception as e:
        await handle_exception(e)


@router.get("/dispensaries/", response_model=List[Dispensary])
async def read_dispensaries(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """Retrieve a list of dispensaries with pagination."""
    try:
        dispensaries = await get_dispensaries(skip=skip, limit=limit)
        return dispensaries
    except Exception as e:
        await handle_exception(e)


@router.get("/dispensaries/{retailer_id}", response_model=Dispensary)
async def read_dispensary(retailer_id: str):
    """Retrieve a specific dispensary by retailer ID."""
    try:
        db_dispensary = await get_dispensary(
            retailer_id
        )  # Ensure this is awaited if it's an async function
        if db_dispensary is None:
            raise HTTPException(status_code=404, detail="Dispensary not found")
        return db_dispensary
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        await handle_exception(e)
