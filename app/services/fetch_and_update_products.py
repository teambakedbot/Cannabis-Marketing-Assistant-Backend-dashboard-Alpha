import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CannMenus API credentials
CANNMENUS_API_KEY = os.getenv("CANNMENUS_API_KEY")
