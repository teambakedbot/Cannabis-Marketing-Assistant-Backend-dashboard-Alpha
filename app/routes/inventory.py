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
from dotenv import load_dotenv


# from fastapi.middleware.throttle import ThrottleMiddleware

# Load environment variables
load_dotenv()

router = APIRouter()


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
