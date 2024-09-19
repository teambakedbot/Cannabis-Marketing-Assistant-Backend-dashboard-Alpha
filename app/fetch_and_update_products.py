import os
import httpx
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CannMenus API credentials
CANNMENUS_API_KEY = os.getenv("CANNMENUS_API_KEY")
