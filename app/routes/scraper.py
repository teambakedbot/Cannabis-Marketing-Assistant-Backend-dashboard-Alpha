from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    status,
)
from ..services.auth_service import (
    get_firebase_user,
)
from ..models.schemas import (
    User,
)
import httpx
import time
from ..utils.firebase_utils import db
from ..config.config import settings, logger
from firebase_admin import firestore

router = APIRouter(prefix="/api/v1")


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


async def scrape_and_store_products(retailer_id: int, user_id: str):
    logger.info(f"Scraping products for retailer_id: {retailer_id}")
    try:
        # Initialize variables for pagination
        page = 1
        all_products = {}
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

                # Group products by SKU
                for product_group in products:
                    sku = product_group.get("sku")
                    if sku not in all_products:
                        # Initialize the parent product with the first product in the group
                        first_product = product_group["products"][0]
                        all_products[sku] = {
                            "retailer_id": retailer_id,
                            "sku": sku,
                            **first_product,
                            "variations": [],
                            "lowest_price": first_product["latest_price"],
                            "percentage_thc": first_product.get("percentage_thc"),
                            "percentage_cbd": first_product.get("percentage_cbd"),
                            "mg_thc": first_product.get("mg_thc"),
                            "mg_cbd": first_product.get("mg_cbd"),
                        }

                    for product in product_group.get("products", []):
                        # Update lowest price if necessary
                        if product["latest_price"] < all_products[sku]["lowest_price"]:
                            all_products[sku]["lowest_price"] = product["latest_price"]

                        # Update THC/CBD values if they're higher
                        for key in [
                            "percentage_thc",
                            "percentage_cbd",
                            "mg_thc",
                            "mg_cbd",
                        ]:
                            if product.get(key) and (
                                not all_products[sku][key]
                                or product[key] > all_products[sku][key]
                            ):
                                all_products[sku][key] = product[key]

                        # Add variation only if it's different from the parent
                        variation = {}
                        for key, value in product.items():
                            if value != all_products[sku].get(key):
                                variation[key] = value

                        if variation:
                            all_products[sku]["variations"].append(variation)

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
        for sku, product in all_products.items():
            if not sku:
                logger.warning(f"Skipping product without SKU: {product}")
                continue
            product_ref = products_ref.document(sku)
            batch.set(
                product_ref,
                {
                    **product,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            updated_product_ids.add(sku)
            count += 1

            # If we've reached the batch limit, commit and reset
            if count % 500 == 0:
                await batch.commit()
                batch = db.batch()

        # Commit any remaining updates
        if count % 500 != 0:
            await batch.commit()

        # Delete products that are no longer associated with this retailer
        query = products_ref.where("retailer_id", "==", retailer_id)
        existing_products = await query.get()

        delete_batch = db.batch()
        delete_count = 0
        for doc in existing_products:
            if doc.id not in updated_product_ids:
                delete_batch.delete(doc.reference)
                delete_count += 1

            # If we've reached the batch limit, commit and reset
            if delete_count % 500 == 0:
                await delete_batch.commit()
                delete_batch = db.batch()

        # Commit any remaining deletes
        if delete_count % 500 != 0:
            await delete_batch.commit()

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


async def scrape_and_store_retailers(user_id: str):
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
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            updated_retailer_ids.add(retailer_id)
            count += 1

            # If we've reached the batch limit, commit and reset
            if count % 500 == 0:
                await batch.commit()
                batch = db.batch()

        # Commit any remaining updates
        if count % 500 != 0:
            await batch.commit()

        logger.info(
            f"Successfully scraped and stored {len(updated_retailer_ids)} retailers"
        )

    except Exception as e:
        logger.error(f"Error occurred while scraping retailers: {e}")
