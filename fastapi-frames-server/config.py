import itertools
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

_config = {
    "BASE_URL": os.environ["BASE_URL"],
    "TUMBLLER_BASE_URLS": {},
    "TUMBLLER_CAMERA_URLS": {},
}

for name, url in itertools.batched(os.environ.get("ROVER_URLS", []), n=2):
    _config["TUMBLLER_BASE_URLS"][name] = f"{url}/tumbllers/{name.lower().replace(' ', '-')}"
    _config["TUMBLLER_CAMERA_URLS"][name] = f"{url}/cameras/{name.lower().replace(' ', '-')}"

# Export configuration variables
BASE_URL = _config["BASE_URL"]
FQDN = os.environ["FQDN"]
TUMBLLER_BASE_URLS = _config["TUMBLLER_BASE_URLS"]
TUMBLLER_CAMERA_URLS = _config["TUMBLLER_CAMERA_URLS"]

FARCASTER_HOSTED_MANIFEST_URL = os.environ["FARCASTER_HOSTED_MANIFEST_URL"]

