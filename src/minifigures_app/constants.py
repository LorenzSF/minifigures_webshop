"""Application constants."""

import os

# Host on which the API runs
HOST = os.getenv("API_HOST", "http://api")

# Port under which the endpoints are exposed
PORT = os.getenv("API_PORT", "8000")

# URL in which the API endpoints are residing
URL = f"{HOST}:{PORT}"
