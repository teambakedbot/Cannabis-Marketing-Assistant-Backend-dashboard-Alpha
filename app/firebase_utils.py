import json
import os
from typing import Dict, Optional
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi import HTTPException
from .config import logger
from datetime import datetime

load_dotenv()


def initialize_firebase():
    """Initialize Firebase using credentials from the environment."""
    if not firebase_admin._apps:
        cred_input = os.getenv("FIREBASE_CREDENTIALS")
        if not cred_input:
            raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

        try:
            cred_json = json.loads(cred_input)
            cred = credentials.Certificate(cred_json)
            logger.debug("Firebase credentials loaded from JSON")
        except json.JSONDecodeError:
            if not os.path.exists(cred_input):
                raise FileNotFoundError(f"Credential path {cred_input} does not exist")
            cred = credentials.Certificate(cred_input)
            logger.debug("Firebase credentials loaded from file: %s", cred_input)

        firebase_admin.initialize_app(cred)
        logger.debug("Firebase initialized successfully")


class FirestoreWrapper:
    def __init__(self, client):
        self.client = client

    def collection(self, *args, **kwargs):
        return CollectionWrapper(self.client.collection(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.client, name)


class CollectionWrapper:
    def __init__(self, collection):
        self.collection = collection

    def document(self, *args, **kwargs):
        return DocumentWrapper(self.collection.document(*args, **kwargs))

    def where(self, *args, **kwargs):
        query = self.collection.where(*args, **kwargs)
        return QueryWrapper(query)

    def get(self, *args, **kwargs):
        # This is a read operation
        results = self.collection.get(*args, **kwargs)
        log_query(
            collection_name=self.collection.id,
            query_type="get",
            query_params={},
        )
        return results

    def stream(self, *args, **kwargs):
        # Log the stream operation
        results = self.collection.stream(*args, **kwargs)
        log_query(
            collection_name=self.collection.id,
            query_type="stream",
            query_params={},
        )
        return results

    # Wrap other collection methods as needed


class DocumentWrapper:
    def __init__(self, document):
        self.document = document

    def get(self, *args, **kwargs):
        # This is a read operation
        result = self.document.get(*args, **kwargs)
        log_query(
            collection_name=self.document._document_path,
            query_type="get",
            query_params={"document_id": self.document.id},
        )
        return result

    def __getattr__(self, name):
        # Pass other methods directly to the base document
        return getattr(self.document, name)


class QueryWrapper:
    def __init__(self, query):
        self.query = query
        self._query_params = []

    def where(self, *args, **kwargs):
        self._query_params.append(("where", args, kwargs))
        new_query = self.query.where(*args, **kwargs)
        return QueryWrapper(new_query)

    def order_by(self, *args, **kwargs):
        self._query_params.append(("order_by", args, kwargs))
        new_query = self.query.order_by(*args, **kwargs)
        return QueryWrapper(new_query)

    def limit(self, *args, **kwargs):
        self._query_params.append(("limit", args, kwargs))
        new_query = self.query.limit(*args, **kwargs)
        return QueryWrapper(new_query)

    def offset(self, *args, **kwargs):
        self._query_params.append(("offset", args, kwargs))
        new_query = self.query.offset(*args, **kwargs)
        return QueryWrapper(new_query)

    def stream(self, *args, **kwargs):
        # This is a read operation
        results = self.query.stream(*args, **kwargs)
        log_query(
            collection_name=self.query._parent.id,
            query_type="stream",
            query_params={"query_params": self._query_params},
        )
        return results

    def get(self, *args, **kwargs):
        # This is a read operation
        results = self.query.get(*args, **kwargs)
        log_query(
            collection_name=self.query._parent.id,
            query_type="get",
            query_params={"query_params": self._query_params},
        )
        return results

    def __getattr__(self, name):
        return getattr(self.query, name)


def log_query(
    collection_name: str, query_type: str, query_params: Optional[Dict] = None
):
    """
    Log a query to Firestore in a dedicated logging collection.

    Parameters:
    - collection_name: The name of the collection being queried.
    - query_type: Type of query (e.g., "where", "get", "stream", etc.).
    - query_params: Additional query parameters (e.g., filters, limits).
    """
    log_entry = {
        "collection": collection_name,
        "query_type": query_type,
        "query_params": query_params or {},
        "timestamp": datetime.utcnow(),
    }

    log_ref = firestore.client().collection("query_logs").document()
    log_ref.set(log_entry)


def verify_firebase_token(token: str):
    """Verify Firebase authentication token."""
    try:
        decoded_token = auth.verify_id_token(token)
        logger.debug("Firebase token successfully verified: %s", decoded_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


initialize_firebase()
db = FirestoreWrapper(firestore.client())
