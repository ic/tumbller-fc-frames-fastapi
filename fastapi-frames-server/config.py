import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Get environment
ENV = os.getenv("ENVIRONMENT", "development")  # Default to development if not set

# Production configuration (for GitHub)
PROD_CONFIG = {
    "TUMBLLER_CAMERA_URLS": {
        "A": "http://rover-a-cam.local/getImage",
        "B": "http://rover-b-cam.local/getImage",
    },
    "BASE_URL": "https://ngrok-ip.ngrok-free.app",
    "TUMBLLER_BASE_URLS": {
        "A": "http://tumbller-a.local",
        "B": "http://tumbller-b.local",
    },
}

# Development configuration (from environment variables)
DEV_CONFIG = {
    "TUMBLLER_CAMERA_URLS": {
        "A": os.getenv("CAMERA_URL_A", PROD_CONFIG["TUMBLLER_CAMERA_URLS"]["A"]),
        "B": os.getenv("CAMERA_URL_B", PROD_CONFIG["TUMBLLER_CAMERA_URLS"]["B"]),
    },
    "BASE_URL": os.getenv("BASE_URL", PROD_CONFIG["BASE_URL"]),
    "TUMBLLER_BASE_URLS": {
        "A": os.getenv("TUMBLLER_URL_A", PROD_CONFIG["TUMBLLER_BASE_URLS"]["A"]),
        "B": os.getenv("TUMBLLER_URL_B", PROD_CONFIG["TUMBLLER_BASE_URLS"]["B"]),
    },
}

# Configuration mapping
CONFIG = {"development": DEV_CONFIG, "production": PROD_CONFIG}

# Get current configuration
current_config = CONFIG[ENV]

# Export configuration variables
TUMBLLER_CAMERA_URLS = current_config["TUMBLLER_CAMERA_URLS"]
FQDN = os.environ["FQDN"]
BASE_URL = current_config["BASE_URL"]
TUMBLLER_BASE_URLS = current_config["TUMBLLER_BASE_URLS"]

FARCASTER_HOSTED_MANIFEST_URL = os.environ["FARCASTER_HOSTED_MANIFEST_URL"]

