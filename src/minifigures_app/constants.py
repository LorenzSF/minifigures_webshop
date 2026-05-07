"""Application constants."""

import os


def _normalize_host(raw_host: str) -> str:
    """Ensure the API host always includes a scheme."""
    if raw_host.startswith(("http://", "https://")):
        return raw_host
    return f"http://{raw_host}"


# Host on which the API runs.
# Default to localhost for local development; container deployments can override
# this with API_HOST=http://api (or a different reachable hostname).
HOST = _normalize_host(os.getenv("API_HOST", "localhost"))

# Port under which the endpoints are exposed
PORT = os.getenv("API_PORT", "8000")

# URL in which the API endpoints are residing
URL = f"{HOST}:{PORT}"
