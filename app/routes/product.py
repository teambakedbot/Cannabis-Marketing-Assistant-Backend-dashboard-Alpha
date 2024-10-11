from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from ..services.recommendation_system import get_search_products
from typing import List, Optional
from ..services.auth_service import (
    get_firebase_user,
)
from ..crud.crud import (
    get_recommended_products,
    search_products,
    get_product,
    update_product,
    delete_product,
    get_products,
)
from ..models.schemas import (
    User,
    Product,
    ProductUpdate,
    ProductResults,
)
import httpx
from ..utils.redis_config import get_redis, FirestoreEncoder
from redis.asyncio import Redis
import json
from ..config.config import settings, logger

router = APIRouter(prefix="/api/v1")


@router.get("/products/", response_model=ProductResults)
async def read_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    redis: Redis = Depends(get_redis),
    retailers: Optional[List[int]] = Query(None, description="List of retailer IDs"),
    states: Optional[List[str]] = Query(None, description="List of states"),
):
    try:
        skip = (page) * limit
        cache_key = f"products:{skip}:{limit}:{','.join(map(str, retailers or []))}:{','.join(states or [])}"
        # cached_products = await redis.get(cache_key)
        # if cached_products:
        #     return json.loads(cached_products)

        results = await get_products(
            skip=skip, limit=limit, retailers=retailers, states=states
        )

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


@router.get("/recommendations/", response_model=List[Product])
async def get_recommendations_endpoint(
    current_user: User = Depends(get_firebase_user),
):
    try:
        return await get_recommended_products(user_id=current_user.id)
    except Exception as e:
        logger.error(f"Error in get_recommendations_endpoint: {str(e)}")
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
