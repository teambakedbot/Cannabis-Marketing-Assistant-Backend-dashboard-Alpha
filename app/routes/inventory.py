from fastapi import (
    APIRouter,
    HTTPException,
)
from typing import List
from ..crud.crud import (
    get_dispensary_inventory,
    create_inventory,
)
from ..models.schemas import (
    Inventory,
    InventoryCreate,
)
from ..config.config import logger

router = APIRouter(
    prefix="/api/v1",
    tags=["inventory"],
    responses={404: {"description": "Not found"}},
)


async def handle_exception(e: Exception) -> HTTPException:
    """Helper function to handle exceptions and log errors."""
    logger.error(f"Error: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/inventory/", response_model=Inventory)
async def create_inventory_endpoint(inventory: InventoryCreate):
    """Create a new inventory item."""
    try:
        return await create_inventory(inventory)
    except Exception as e:
        await handle_exception(e)


@router.get("/inventory/{retailer_id}", response_model=List[Inventory])
async def read_dispensary_inventory_endpoint(retailer_id: str):
    """Retrieve the inventory for a specific dispensary by retailer ID."""
    try:
        inventory = await get_dispensary_inventory(retailer_id)
        return inventory
    except Exception as e:
        await handle_exception(e)
