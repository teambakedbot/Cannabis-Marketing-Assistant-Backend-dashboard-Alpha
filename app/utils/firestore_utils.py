from typing import Any, Optional, TypeVar, Callable
from functools import wraps
import asyncio
from google.cloud import firestore
from google.api_core import retry
from ..config.config import logger

T = TypeVar("T")


class FirestoreError(Exception):
    """Base class for Firestore-related errors."""

    pass


class DocumentNotFoundError(FirestoreError):
    """Raised when a document is not found."""

    pass


class PermissionError(FirestoreError):
    """Raised when there are permission issues."""

    pass


def with_firestore_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    maximum_delay: float = 5.0,
    retry_on_exceptions: tuple = (Exception,),
):
    """
    Decorator that implements retry logic for Firestore operations.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        maximum_delay: Maximum delay between retries in seconds
        retry_on_exceptions: Tuple of exceptions to retry on
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Operation failed after {max_retries} attempts: {str(e)}"
                        )
                        raise

                    delay = min(initial_delay * (2**attempt), maximum_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay} seconds: {str(e)}"
                    )
                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class FirestoreClient:
    """Wrapper class for Firestore operations with retry logic."""

    def __init__(self, db: firestore.AsyncClient):
        self.db = db

    @with_firestore_retry()
    async def get_document(
        self, collection: str, document_id: str
    ) -> Optional[firestore.DocumentSnapshot]:
        """
        Get a document from Firestore with retry logic.

        Args:
            collection: Collection name
            document_id: Document ID

        Returns:
            Document snapshot or None if not found
        """
        doc_ref = self.db.collection(collection).document(document_id)
        try:
            doc = await doc_ref.get()
            if not doc.exists:
                raise DocumentNotFoundError(
                    f"Document {document_id} not found in collection {collection}"
                )
            return doc
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise

    @with_firestore_retry()
    async def set_document(
        self, collection: str, document_id: str, data: dict, merge: bool = False
    ) -> None:
        """
        Set a document in Firestore with retry logic.

        Args:
            collection: Collection name
            document_id: Document ID
            data: Document data
            merge: Whether to merge with existing data
        """
        doc_ref = self.db.collection(collection).document(document_id)
        try:
            await doc_ref.set(data, merge=merge)
        except Exception as e:
            logger.error(f"Error setting document {document_id}: {str(e)}")
            raise

    @with_firestore_retry()
    async def update_document(
        self, collection: str, document_id: str, data: dict
    ) -> None:
        """
        Update a document in Firestore with retry logic.

        Args:
            collection: Collection name
            document_id: Document ID
            data: Update data
        """
        doc_ref = self.db.collection(collection).document(document_id)
        try:
            await doc_ref.update(data)
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            raise

    @with_firestore_retry()
    async def delete_document(self, collection: str, document_id: str) -> None:
        """
        Delete a document from Firestore with retry logic.

        Args:
            collection: Collection name
            document_id: Document ID
        """
        doc_ref = self.db.collection(collection).document(document_id)
        try:
            await doc_ref.delete()
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    @with_firestore_retry()
    async def query_collection(
        self,
        collection: str,
        filters: Optional[list] = None,
        order_by: Optional[tuple] = None,
        limit: Optional[int] = None,
    ) -> list:
        """
        Query a collection with filters and retry logic.

        Args:
            collection: Collection name
            filters: List of tuples (field, operator, value)
            order_by: Tuple of (field, direction)
            limit: Maximum number of documents to return

        Returns:
            List of document snapshots
        """
        query = self.db.collection(collection)

        if filters:
            for field, op, value in filters:
                query = query.where(field, op, value)

        if order_by:
            field, direction = order_by
            query = query.order_by(field, direction=direction)

        if limit:
            query = query.limit(limit)

        try:
            docs = await query.get()
            return [doc for doc in docs]
        except Exception as e:
            logger.error(f"Error querying collection {collection}: {str(e)}")
            raise


# Create a global Firestore client instance
firestore_client = FirestoreClient(db)
