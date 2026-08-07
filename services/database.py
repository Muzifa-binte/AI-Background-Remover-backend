"""
MongoDB connection and helper utilities powered by Motor (async driver).
"""

import os
from urllib.parse import quote_plus
from motor.motor_asyncio import AsyncIOMotorClient

client: AsyncIOMotorClient = None  # type: ignore


async def connect_db() -> None:
    """Open the MongoDB connection. Call from FastAPI startup event."""
    global client
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    # If URI contains unencoded special chars in password, encode them
    # Motor handles the full URI string directly
    client = AsyncIOMotorClient(mongo_uri)


def get_db_name() -> str:
    return os.getenv("MONGO_DB_NAME", "ai_bg_remover")


async def close_db() -> None:
    """Close the MongoDB connection. Call from FastAPI shutdown event."""
    if client:
        client.close()


def get_collection(name: str):
    """Return a Motor collection by name."""
    return client[get_db_name()][name]
