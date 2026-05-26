"""
Common utilities shared across ai-builds projects.
"""

import os
from dotenv import load_dotenv


def load_env(env_file: str = ".env") -> None:
    """Load environment variables from a .env file."""
    load_dotenv(env_file)


def get_env(key: str, required: bool = True) -> str:
    """Get an environment variable, raising an error if required and missing."""
    value = os.getenv(key)
    if required and not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value
