from pathlib import Path
import os

def project_root() -> Path:
    # Inside Docker we set WORKDIR /app, and your code lives /app/app
    # Using an env var gives you full control in prod.
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]  # /app
