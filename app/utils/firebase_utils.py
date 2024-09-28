import json
import os
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud.firestore_v1.async_client import AsyncClient
from google.oauth2 import service_account
from fastapi import HTTPException
from ..config.config import logger, settings


db = None


def initialize_firebase():
    global db
    """Initialize Firebase using credentials from the environment."""
    if not firebase_admin._apps:
        cred_input = settings.FIREBASE_CREDENTIALS
        if not cred_input:
            raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

        try:
            # Attempt to load credentials from JSON string
            cred_json = json.loads(cred_input)
            # Initialize Firebase Admin SDK with the credential dict
            firebase_admin.initialize_app(credentials.Certificate(cred_json))
            logger.debug("Firebase credentials loaded from JSON")
            # Create service_account_credentials from the dict for AsyncClient
            service_account_credentials = (
                service_account.Credentials.from_service_account_info(cred_json)
            )
        except json.JSONDecodeError:
            # If JSON parsing fails, treat cred_input as a file path
            if not os.path.exists(cred_input):
                raise FileNotFoundError(f"Credential path {cred_input} does not exist")
            # Initialize Firebase Admin SDK with the credential file
            firebase_admin.initialize_app(credentials.Certificate(cred_input))
            logger.debug("Firebase credentials loaded from file: %s", cred_input)
            # Create service_account_credentials from the file for AsyncClient
            service_account_credentials = (
                service_account.Credentials.from_service_account_file(cred_input)
            )

        # Initialize AsyncClient with the same credentials and project ID
        db = AsyncClient(
            credentials=service_account_credentials,
            project=service_account_credentials.project_id,
        )
        logger.debug("Firebase initialized successfully")


async def verify_firebase_token(token: str):
    """Verify Firebase authentication token."""
    try:
        decoded_token = auth.verify_id_token(token)
        logger.debug("Firebase token successfully verified: %s", decoded_token)
        return decoded_token
    except Exception as e:
        logger.error(f"Error in verify_firebase_token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")


initialize_firebase()
