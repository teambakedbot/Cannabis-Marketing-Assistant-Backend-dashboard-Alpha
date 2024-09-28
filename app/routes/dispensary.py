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

router = APIRouter()


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
